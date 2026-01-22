# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import wandb
wandb.login(key="xxx")

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class RewardManager():
    """The reward manager with R_align support."""
    
    def __init__(self, tokenizer, num_examine, format_score=0., eval_metrics=None,
                 r_align_weight=0.1, c3_weight=1, semantic_weight=0,
                 r_align_only_correct=False, max_combined_reward=2.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.eval_metrics = eval_metrics or {'em': True}
        
        self.r_align_weight = r_align_weight
        self.c3_weight = c3_weight
        self.semantic_weight = semantic_weight
        self.r_align_only_correct = r_align_only_correct  
        self.max_combined_reward = max_combined_reward  
        
        enabled = [k for k, v in self.eval_metrics.items() if v]
        self.train_metric = enabled[0] if len(enabled) == 1 else None
        self.is_training = self.train_metric is not None
        self.uid_check_done = False

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
                
        if self.is_training:
            return self._compute_single_metric(data, self.train_metric)
        else:
            results = {}
            for metric_name in self.eval_metrics:
                if self.eval_metrics[metric_name]:
                    results[metric_name] = self._compute_single_metric(data, metric_name)
            return results

    def _extract_and_normalize_answers(self, data: DataProto):
        from verl.utils.reward_score.qa_em import extract_solution, normalize_answer
        
        answers = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            answer = extract_solution(sequences_str)
            if answer is None:
                answer = ""
            normalized_answer = normalize_answer(answer)
            answers.append(normalized_answer)
        
        return answers
    
    def _compute_character_3_similarity(self, answer1, answer2):
        def get_3gram(text):
            if len(text) < 3:
                return set()
            return set([text[i:i+3] for i in range(len(text)-2)])
        
        gram1 = get_3gram(answer1)
        gram2 = get_3gram(answer2)
        
        if len(gram1) == 0 or len(gram2) == 0:
            return 0.0
        
        recall_1_to_2 = len(gram1 & gram2) / len(gram1) if len(gram1) > 0 else 0.0
        recall_2_to_1 = len(gram1 & gram2) / len(gram2) if len(gram2) > 0 else 0.0
        
        return (recall_1_to_2 + recall_2_to_1) / 2.0
    
    def _compute_r_align(self, data: DataProto, original_rewards):
        """
        Anti-consistency version of R_align:
        - Penalizes similarity between incorrect answers.
        """
        answers = self._extract_and_normalize_answers(data)
        uids = data.non_tensor_batch.get('uid', None)
        n = len(answers)
        r_align_scores = np.zeros(n, dtype=np.float32)
        
        if uids is None:
            return r_align_scores
        
        uid_to_indices = defaultdict(list)
        for i, uid in enumerate(uids):
            uid_to_indices[uid].append(i)
        
        groups_processed = 0
        bad_threshold = 0.4   
        margin = 0.5          
        
        for uid, indices in uid_to_indices.items():
            if len(indices) <= 1:
                continue
            
            rewards_in_group = [original_rewards[i] for i in indices]
            bad_indices = [i for i in indices if original_rewards[i] < bad_threshold]
            
            if len(bad_indices) <= 1:
                continue
            
            groups_processed += 1
            group_answers = [answers[i] for i in bad_indices]
            
            k = len(bad_indices)
            c3_sim_matrix = np.zeros((k, k), dtype=np.float32)
            for i_local in range(k):
                for j_local in range(k):
                    if i_local == j_local:
                        c3_sim_matrix[i_local, j_local] = 1.0
                    else:
                        c3_sim_matrix[i_local, j_local] = self._compute_character_3_similarity(
                            group_answers[i_local], group_answers[j_local]
                        )
            
            combined_sim_matrix = (self.c3_weight * c3_sim_matrix)
            
            for local_i, global_i in enumerate(bad_indices):
                other_sims = [combined_sim_matrix[local_i, local_j]
                            for local_j in range(k) if local_j != local_i]
                if not other_sims:
                    continue
                
                max_sim = float(np.max(other_sims))
                penalty_raw = max(0.0, max_sim - margin)
                
                if penalty_raw <= 0.0:
                    continue
                
                group_weight = min(1.0, len(bad_indices) / 5.0)
                r_align_scores[global_i] = - penalty_raw * group_weight
        
        return r_align_scores
    
    def _compute_single_metric(self, data: DataProto, metric_name: str):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        original_rewards = []
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = self._select_rm_score_fn(data_source, metric_name)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                    format_score=self.format_score)
            reward_tensor[i, valid_response_length - 1] = score
            original_rewards.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"=== {metric_name.upper()} Evaluation ===")
                print(f"Score: {score}")
                print(sequences_str[:200])
        
        if self.is_training and self.r_align_weight > 0:
            r_align_scores = self._compute_r_align(data, np.array(original_rewards))
            r_align_scores = np.clip(r_align_scores, -0.5, 0.0)
            
            for i in range(len(data)):
                data_item = data[i]
                prompt_length = data_item.batch['prompts'].shape[-1]
                valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
                
                combined_reward = original_rewards[i] + self.r_align_weight * r_align_scores[i]
                combined_reward = max(0.0, min(combined_reward, self.max_combined_reward))
                
                reward_tensor[i, valid_response_length - 1] = combined_reward

        return reward_tensor

    def _select_rm_score_fn(self, data_source, metric_name):
        if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'mkqa']:
            if metric_name == 'em':
                return qa_em.compute_score_em
            elif metric_name == 'fem':
                return qa_em.compute_score_fem
            elif metric_name == 'c3recall':
                return qa_em.compute_score_c3recall
            else:
                raise NotImplementedError(f"Metric {metric_name} not implemented")
        else:
            raise NotImplementedError(f"Data source {data_source} not implemented")

import ray
import hydra

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))

@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from pprint import pprint
    from omegaconf import OmegaConf
    
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0, 
        eval_metrics={'c3recall': True},
        r_align_weight=0.02,   
        c3_weight=1.0,         
        semantic_weight=0.0,   
        max_combined_reward=1.5,
    )

    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, eval_metrics={'em': True})

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()

if __name__ == '__main__':
    main()