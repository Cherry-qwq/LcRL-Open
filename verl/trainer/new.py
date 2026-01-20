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
wandb.login(key="c37e672252e2c2621f7a3d03a8aad128c8dcb474")

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class DocumentSimilarityCalculator:
    """计算文档和答案之间的语义相似度"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = "cuda"):
        """
        初始化相似度计算器
        
        Args:
            model_name: multilingual-e5模型名称
            device: 运行设备
        """
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        print(f"[INFO] E5 calculator initialized (model will load on first use)")
    def _ensure_model_loaded(self):
        """延迟加载模型（第一次调用时）"""
        if self.model is not None:
            return
            
        print(f"[INFO] Loading multilingual-e5 model: {self.model_name}")
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # 尝试GPU，失败则CPU
            if self.device.startswith("cuda"):
                try:
                    self.model = self.model.to(self.device)
                    print(f"[INFO] E5 loaded on {self.device}")
                except:
                    print(f"[WARNING] GPU not available, using CPU for E5")
                    self.model = self.model.to("cpu")
                    self.device = "cpu"
            else:
                self.model = self.model.to(self.device)
                print(f"[INFO] E5 loaded on {self.device}")
                
            self.model.eval()
            print(f"[INFO] Multilingual-e5 model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load E5: {e}")
            raise
    def get_embeddings(self, texts: list, prefix: str = "query: ") -> torch.Tensor:
        """
        获取文本的embeddings
        
        Args:
            texts: 文本列表
            prefix: E5模型需要的前缀（"query: " 或 "passage: "）
            
        Returns:
            embeddings tensor [batch_size, hidden_dim]
        """
        self._ensure_model_loaded() 
        # 添加E5模型需要的前缀
        texts_with_prefix = [prefix + text for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts_with_prefix,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        # 获取embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用mean pooling
            embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            # 归一化
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def compute_similarity(self, answer: str, documents: list) -> float:
        """
        计算答案和文档集合的相似度
        
        Args:
            answer: 答案文本
            documents: 文档列表
            
        Returns:
            相似度分数 (0-1之间)
        """
        if not documents or not answer.strip():
            return 0.0
        
        try:
            # 清理文档文本
            clean_docs = []
            for doc in documents:
                # 移除language prefix
                doc_clean = re.sub(r'^\[.*?\]\s*', '', doc)
                # 移除Doc标记
                doc_clean = re.sub(r'^Doc \d+\(Title:.*?\)\s*', '', doc_clean)
                doc_clean = doc_clean.strip()
                if doc_clean:
                    clean_docs.append(doc_clean)
            
            if not clean_docs:
                return 0.0
            
            # 获取answer的embedding (作为query)
            answer_embedding = self.get_embeddings([answer], prefix="query: ")
            
            # 获取documents的embeddings (作为passages)
            doc_embeddings = self.get_embeddings(clean_docs, prefix="passage: ")
            
            # 计算余弦相似度
            similarities = torch.mm(answer_embedding, doc_embeddings.T).squeeze(0)
            
            # 使用最大相似度（表示答案至少与某个文档高度相关）
            max_similarity = similarities.max().item()
            
            # 也可以使用平均相似度（表示答案整体与文档集的相关性）
            avg_similarity = similarities.mean().item()
            
            # 这里我们使用加权组合：70%最大相似度 + 30%平均相似度
            # 这样既鼓励答案与最相关文档对齐，也考虑整体文档质量
            combined_similarity = 0.7 * max_similarity + 0.3 * avg_similarity
            
            return combined_similarity
            
        except Exception as e:
            print(f"[ERROR] Failed to compute similarity: {e}")
            return 0.0


class RewardManager():
    """The reward manager with document similarity bonus."""
    
    def __init__(self, tokenizer, num_examine, format_score=0., eval_metrics=None, 
                 use_doc_similarity=True, similarity_weight=0.2, device="cuda") -> None:
        """
        Args:
            tokenizer: 分词器
            num_examine: 打印样例数量
            format_score: 格式分数
            eval_metrics: 评估指标
            use_doc_similarity: 是否使用文档相似度
            similarity_weight: 文档相似度权重（建议0.1-0.3）
            device: 运行设备
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.eval_metrics = eval_metrics or {'em': True}
        
        # 文档相似度相关
        self.use_doc_similarity = use_doc_similarity
        self.similarity_weight = similarity_weight
        
        # 训练模式判断
        enabled = [k for k, v in self.eval_metrics.items() if v]
        self.train_metric = enabled[0] if len(enabled) == 1 else None
        self.is_training = self.train_metric is not None
        
        # 打印配置信息
        print(f"\n{'='*80}")
        print(f"[REWARD CONFIG] Initializing RewardManager")
        print(f"[REWARD CONFIG] Eval metrics: {self.eval_metrics}")
        print(f"[REWARD CONFIG] Training metric: {self.train_metric}")
        print(f"[REWARD CONFIG] Is training mode: {self.is_training}")
        print(f"[REWARD CONFIG] Use doc similarity: {self.use_doc_similarity}")
        print(f"[REWARD CONFIG] Similarity weight: {self.similarity_weight}")
        print(f"[REWARD CONFIG] Device: {device}")
        print(f"[REWARD CONFIG] Num examine: {self.num_examine}")
        print(f"{'='*80}\n")
        
        if self.use_doc_similarity:
            self.similarity_calculator = DocumentSimilarityCalculator(device=device)
            print(f"[INFO] Document similarity enabled with weight={similarity_weight} (E5 will load on first use)")
        else:
            self.similarity_calculator = None
            print(f"[INFO] Document similarity disabled")

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

    def _extract_answer(self, sequences_str: str) -> str:
        """从生成的序列中提取答案部分"""
        # 提取<answer>标签之间的内容
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, sequences_str, re.DOTALL)
        
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # 如果没有answer标签，尝试提取最后一部分作为答案
            # 移除information标签
            clean_str = re.sub(r'<information>.*?</information>', '', sequences_str, flags=re.DOTALL)
            # 移除search标签
            clean_str = re.sub(r'<search>.*?</search>', '', clean_str, flags=re.DOTALL)
            answer = clean_str.strip()
        
        return answer

    def _compute_single_metric(self, data: DataProto, metric_name: str):
        """计算单个指标，包含文档相似度bonus"""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        
        # 统计信息 - 追踪所有样本
        similarity_scores = []
        base_scores = []
        bonus_scores = []
        samples_with_docs = 0
        samples_without_docs = 0
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # 计算基础reward (c3recall或其他指标)
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = self._select_rm_score_fn(data_source, metric_name)
            base_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                         format_score=self.format_score)
            base_scores.append(base_score)
            
            # 计算文档相似度bonus
            doc_similarity_bonus = 0.0
            similarity = 0.0
            
            if self.use_doc_similarity and self.is_training:
                # 从meta_info中获取检索到的文档（避免chunk问题）
                retrieved_docs = data.meta_info.get('retrieved_docs', [[]])[i]
                
                if retrieved_docs and len(retrieved_docs) > 0:
                    samples_with_docs += 1
                    # 提取答案
                    answer = self._extract_answer(sequences_str)
                    
                    if answer:
                        # 计算相似度
                        similarity = self.similarity_calculator.compute_similarity(answer, retrieved_docs)
                        doc_similarity_bonus = self.similarity_weight * similarity
                        similarity_scores.append(similarity)
                    else:
                        # 无法提取答案
                        similarity_scores.append(0.0)
                else:
                    samples_without_docs += 1
                    similarity_scores.append(0.0)
            
            bonus_scores.append(doc_similarity_bonus)
            
            # 组合最终reward
            final_score = base_score + doc_similarity_bonus
            reward_tensor[i, valid_response_length - 1] = final_score

            # 打印示例
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n{'='*80}")
                print(f"=== {metric_name.upper()} Evaluation (Sample {i}) ===")
                print(f"Data source: {data_source}")
                print(f"Base score ({metric_name}): {base_score:.4f}")
                if self.use_doc_similarity and self.is_training:
                    print(f"Similarity weight: {self.similarity_weight}")
                    print(f"Raw similarity: {similarity:.4f}")
                    print(f"Doc similarity bonus: {doc_similarity_bonus:.4f}")
                    print(f"Retrieved docs count: {len(retrieved_docs) if retrieved_docs else 0}")
                    print(f"Final score: {final_score:.4f}")
                print(f"\nGenerated sequence:")
                print(sequences_str[:500] + "..." if len(sequences_str) > 500 else sequences_str)
                print(f"{'='*80}\n")
        
        # 打印统计信息 - 总是打印（如果启用了文档相似度）
        if self.use_doc_similarity and self.is_training:
            print(f"\n{'='*80}")
            print(f"[REWARD STATS] Batch size: {len(data)}")
            print(f"[REWARD STATS] Similarity weight: {self.similarity_weight}")
            print(f"[REWARD STATS] Samples with docs: {samples_with_docs}, without docs: {samples_without_docs}")
            print(f"[REWARD STATS] Base {metric_name} - Mean: {np.mean(base_scores):.4f}, "
                  f"Std: {np.std(base_scores):.4f}, Min: {np.min(base_scores):.4f}, Max: {np.max(base_scores):.4f}")
            
            if len(similarity_scores) > 0:
                print(f"[REWARD STATS] Doc Similarity - Mean: {np.mean(similarity_scores):.4f}, "
                      f"Std: {np.std(similarity_scores):.4f}, Min: {np.min(similarity_scores):.4f}, Max: {np.max(similarity_scores):.4f}")
                print(f"[REWARD STATS] Bonus - Mean: {np.mean(bonus_scores):.4f}, "
                      f"Std: {np.std(bonus_scores):.4f}")
            else:
                print(f"[REWARD STATS] Doc Similarity - No valid similarity scores computed")
            
            nonzero_rewards = reward_tensor[reward_tensor != 0]

            print(f"[REWARD STATS] Final Reward - Mean: {nonzero_rewards.mean():.4f}, "
                f"Std: {nonzero_rewards.std():.4f}, Min: {nonzero_rewards.min():.4f}, Max: {nonzero_rewards.max():.4f}")
            print(f"{'='*80}\n")

        return reward_tensor
    
    def _select_rm_score_fn(self, data_source, metric_name):
        """根据数据源和指标名称选择评分函数"""
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
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
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

    # 训练时使用c3recall + 文档相似度
    # 参数说明：
    # - use_doc_similarity=True: 启用文档相似度
    # - similarity_weight=0.2: 相似度权重（可调整，建议0.1-0.3）
    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0, 
        eval_metrics={'c3recall': True},
        use_doc_similarity=True,
        similarity_weight=0.07,  # 可根据实验调整
        device="cuda"
    )

    # 验证时只使用em，不加文档相似度bonus
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0, 
        eval_metrics={'em': True},
        use_doc_similarity=False  # 验证时不使用相似度
    )

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