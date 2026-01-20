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
from verl.trainer.ppo.ray_trainer_parallel import RayPPOTrainer
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
# def _select_rm_score_fn(data_source):
#     if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'mkqa']:
#         return qa_em.compute_score_em
#     else:
#         raise NotImplementedError


# class RewardManager_ori():
#     """The reward manager.
#     """

#     def __init__(self, tokenizer, num_examine, format_score=0., eval_metrics=None)  -> None:
#         self.tokenizer = tokenizer
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.format_score = format_score
#         self.eval_metrics = eval_metrics or {'em': True}

#     def __call__(self, data: DataProto):
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if 'rm_scores' in data.batch.keys():
#             return data.batch['rm_scores']

#         reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

#         # all_scores = []

#         already_print_data_sources = {}

#         for i in range(len(data)):
#             data_item = data[i]  # DataProtoItem

#             prompt_ids = data_item.batch['prompts']

#             prompt_length = prompt_ids.shape[-1]

#             valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            # response_ids = data_item.batch['responses']
            # valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            # valid_response_ids = response_ids[:valid_response_length]

#             # decode
#             sequences = torch.cat((valid_prompt_ids, valid_response_ids))
#             sequences_str = self.tokenizer.decode(sequences)

#             ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

#             # select rm_score
#             data_source = data_item.non_tensor_batch['data_source']
#             compute_score_fn = _select_rm_score_fn(data_source)

#             score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

#             reward_tensor[i, valid_response_length - 1] = score
#             # all_scores.append(score)

#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print(sequences_str)
        
#         # print(f"[DEBUG] all_scores: {all_scores}")
#         # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
#         # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
#         # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
#         # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
#         # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

#         return reward_tensor
class RewardManager():
    """The reward manager with R_align support."""
    
    def __init__(self, tokenizer, num_examine, format_score=0., eval_metrics=None,
                 r_align_weight=0.1, c3_weight=1, semantic_weight=0,
                 r_align_only_correct=False, max_combined_reward=2.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.eval_metrics = eval_metrics or {'em': True}
        
        # R_align相关参数
        self.r_align_weight = r_align_weight
        self.c3_weight = c3_weight
        self.semantic_weight = semantic_weight
        self.r_align_only_correct = r_align_only_correct  # 是否只对正确答案加R_align
        self.max_combined_reward = max_combined_reward  # 最大reward上限
        
        # 加载multilingual-e5模型
        # try:
        #     self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
        #     print("[INFO] Loaded multilingual-e5-base model on CPU for R_align")
        # except Exception as e:
        #     print(f"[WARNING] Failed to load embedding model: {e}")
        #     self.embedding_model = None
        
        enabled = [k for k, v in self.eval_metrics.items() if v]
        self.train_metric = enabled[0] if len(enabled) == 1 else None
        self.is_training = self.train_metric is not None
        
        # 用于检查uid是否有效
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
        """提取并标准化答案（用于R_align计算）"""
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
            
            # 提取答案
            answer = extract_solution(sequences_str)
            if answer is None:
                answer = ""
            # 标准化答案
            normalized_answer = normalize_answer(answer)
            answers.append(normalized_answer)
        
        return answers
    
    def _compute_character_3_similarity(self, answer1, answer2):
        """计算character-3-recall相似度"""
        def get_3gram(text):
            if len(text) < 3:
                return set()
            return set([text[i:i+3] for i in range(len(text)-2)])
        
        gram1 = get_3gram(answer1)
        gram2 = get_3gram(answer2)
        
        if len(gram1) == 0 or len(gram2) == 0:
            return 0.0
        
        # 对称的recall（双向平均）
        recall_1_to_2 = len(gram1 & gram2) / len(gram1) if len(gram1) > 0 else 0.0
        recall_2_to_1 = len(gram1 & gram2) / len(gram2) if len(gram2) > 0 else 0.0
        
        return (recall_1_to_2 + recall_2_to_1) / 2.0
    
    # def _compute_semantic_similarity(self, answers):
    #     """使用multilingual-e5计算语义相似度矩阵（在CPU上）"""
    #     if len(answers) == 0 or self.embedding_model is None:
    #         return np.eye(len(answers))
        
    #     # 添加e5的instruction prefix
    #     answers_with_prefix = [f"query: {ans}" for ans in answers]
        
    #     try:
    #         # 获取embeddings（在CPU上）
    #         embeddings = self.embedding_model.encode(
    #             answers_with_prefix, 
    #             convert_to_tensor=True,
    #             device='cuda',  # 显式指定CPU
    #             show_progress_bar=False
    #         )
            
    #         # 计算cosine相似度矩阵
    #         from torch.nn.functional import cosine_similarity
    #         n = len(answers)
    #         similarity_matrix = torch.zeros((n, n))
            
    #         for i in range(n):
    #             for j in range(n):
    #                 if i == j:
    #                     similarity_matrix[i, j] = 1.0
    #                 else:
    #                     sim = cosine_similarity(embeddings[i].unsqueeze(0), 
    #                                           embeddings[j].unsqueeze(0))
    #                     similarity_matrix[i, j] = sim.item()
            
    #         return similarity_matrix.cpu().numpy()
            
    #     except Exception as e:
    #         print(f"[WARNING] Semantic similarity computation failed: {e}, using identity matrix")
    #         return np.eye(len(answers))
    
    # def _compute_r_align_v1(self, data: DataProto):
    #     """计算跨样本答案一致性奖励（R_align）"""
    #     # 提取并标准化所有答案
    #     answers = self._extract_and_normalize_answers(data)
    #     n = len(answers)
        
    #     # 获取每个样本的uid（用于分组同一问题的多条采样）
    #     uids = data.non_tensor_batch.get('uid', None)
        
    #     # ✅ 修复1：检查uid是否存在且有效
    #     if uids is None:
    #         print("[WARNING] 'uid' not found in non_tensor_batch! R_align will be 0. "
    #               "Make sure your data loading adds 'uid' field.")
    #         return np.zeros(n)
        
    #     # 检查是否有重复的uid（同一问题的多条采样）
    #     if not self.uid_check_done:
    #         unique_uids = len(set(uids))
    #         if unique_uids == n:
    #             print(f"[WARNING] All {n} samples have unique uids! R_align requires multiple "
    #                   f"samples per question. Check if you're using repeat() in your code.")
    #         else:
    #             print(f"[INFO] R_align: {n} samples grouped into {unique_uids} questions. "
    #                   f"Average {n/unique_uids:.1f} samples per question.")
    #         self.uid_check_done = True
        
    #     # 按uid分组
    #     uid_to_indices = defaultdict(list)
    #     for i, uid in enumerate(uids):
    #         uid_to_indices[uid].append(i)
        
    #     # 初始化R_align scores
    #     r_align_scores = np.zeros(n)
        
    #     # 统计信息
    #     groups_with_multiple = 0
        
    #     # 对每个问题组计算R_align
    #     for uid, indices in uid_to_indices.items():
    #         if len(indices) <= 1:
    #             # 只有一个采样，R_align为0
    #             continue
            
    #         groups_with_multiple += 1
    #         group_answers = [answers[i] for i in indices]
            
    #         # 1. 计算character-3-recall相似度矩阵
    #         c3_sim_matrix = np.zeros((len(indices), len(indices)))
    #         for i in range(len(indices)):
    #             for j in range(len(indices)):
    #                 if i == j:
    #                     c3_sim_matrix[i, j] = 1.0
    #                 else:
    #                     c3_sim_matrix[i, j] = self._compute_character_3_similarity(
    #                         group_answers[i], group_answers[j]
    #                     )
            
    #         # 2. 计算语义相似度矩阵
    #         semantic_sim_matrix = self._compute_semantic_similarity(group_answers)
            
    #         # 3. 加权融合两种相似度
    #         combined_sim_matrix = (self.c3_weight * c3_sim_matrix + 
    #                               self.semantic_weight * semantic_sim_matrix)
            
    #         # 4. 计算每个样本的R_align（与其他样本的平均相似度）
    #         for local_i, global_i in enumerate(indices):
    #             # 排除自己，计算与其他样本的平均相似度
    #             other_sims = [combined_sim_matrix[local_i, local_j] 
    #                         for local_j in range(len(indices)) if local_j != local_i]
    #             r_align_scores[global_i] = np.mean(other_sims) if other_sims else 0.0
        
    #     if groups_with_multiple > 0:
    #         print(f"[R_align] Computed for {groups_with_multiple} question groups")
        
    #     return r_align_scores
    # 方案1A：只计算正确答案之间的R_align


    # def _compute_single_metric_v1(self, data: DataProto, metric_name: str):
    #     """计算单个指标（包含R_align融合）"""
    #     reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    #     already_print_data_sources = {}
        
    #     # 存储原始reward，用于后续判断是否正确
    #     original_rewards = []
        
    #     for i in range(len(data)):
    #         data_item = data[i]
    #         prompt_ids = data_item.batch['prompts']
    #         prompt_length = prompt_ids.shape[-1]
    #         valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            # response_ids = data_item.batch['responses']
            # valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            # valid_response_ids = response_ids[:valid_response_length]

    #         sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    #         sequences_str = self.tokenizer.decode(sequences)
    #         ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

    #         data_source = data_item.non_tensor_batch['data_source']
    #         compute_score_fn = self._select_rm_score_fn(data_source, metric_name)
    #         score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
    #                                 format_score=self.format_score)
    #         reward_tensor[i, valid_response_length - 1] = score
    #         original_rewards.append(score)

    #         if data_source not in already_print_data_sources:
    #             already_print_data_sources[data_source] = 0
    #         if already_print_data_sources[data_source] < self.num_examine:
    #             already_print_data_sources[data_source] += 1
    #             print(f"=== {metric_name.upper()} Evaluation ===")
    #             print(f"Data source: {data_source}")
    #             print(f"Score: {score}")
    #             print(sequences_str)
        
    #     # 计算R_align并融合到总reward（仅在训练时）
    #     if self.is_training and self.r_align_weight > 0:
    #         r_align_scores = self._compute_r_align(data)
            
    #         # 统计信息
    #         num_with_align = np.sum(r_align_scores > 0)
            
    #         # 将R_align添加到最后一个token的reward
    #         for i in range(len(data)):
    #             data_item = data[i]
    #             prompt_ids = data_item.batch['prompts']
    #             prompt_length = prompt_ids.shape[-1]
    #             response_ids = data_item.batch['responses']
                ## valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                # valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())

                
    #             original_reward = original_rewards[i]
                
    #             # ✅ 修复2：根据策略决定是否加R_align
    #             if self.r_align_only_correct:
    #                 # 策略1：只对正确答案加R_align（更保守）
    #                 if original_reward > 0.5:  # 认为答案正确
    #                     combined_reward = original_reward + self.r_align_weight * r_align_scores[i]
    #                 else:
    #                     combined_reward = original_reward
    #             else:
    #                 # 策略2：对所有答案加R_align（包括错误答案）
    #                 combined_reward = original_reward + self.r_align_weight * r_align_scores[i]
                
    #             # ✅ 修复3：只clip下限，允许奖励超过1.0
    #             # 这样正确且一致的答案可以获得>1.0的奖励
    #             combined_reward = np.clip(combined_reward, 0.0, self.max_combined_reward)
                
    #             reward_tensor[i, valid_response_length - 1] = combined_reward
            
    #         # 打印统计信息
    #         if num_with_align > 0:
    #             valid_r_align = r_align_scores[r_align_scores > 0]
    #             print(f"[R_align] Samples with R_align: {num_with_align}/{len(r_align_scores)}")
    #             print(f"[R_align] Mean: {np.mean(valid_r_align):.4f}, "
    #                   f"Max: {np.max(valid_r_align):.4f}, "
    #                   f"Min: {np.min(valid_r_align):.4f}")

    #     return reward_tensor
    # 改进方案：使用相对R_align，避免"坍塌到错误一致性"

    # def _compute_r_align_v2(self, data: DataProto, original_rewards):
    #     """
    #     改进的R_align计算：
    #     1. 只在正确答案间计算一致性
    #     2. 使用相对奖励（减去均值）保持奖励分布
    #     3. 根据正确答案数量调整权重
    #     """
    #     answers = self._extract_and_normalize_answers(data)
    #     uids = data.non_tensor_batch.get('uid', None)
    #     n = len(answers)
    #     r_align_scores = np.zeros(n)
        
    #     if uids is None:
    #         return r_align_scores
        
    #     uid_to_indices = defaultdict(list)
    #     for i, uid in enumerate(uids):
    #         uid_to_indices[uid].append(i)
        
    #     groups_processed = 0
    #     bad_threshold = 0.4   # c3recall < 0.4 视为“明显错误”
    #     margin = 0.5
        
    #     for uid, indices in uid_to_indices.items():
    #         if len(indices) <= 1:
    #             continue
            
    #         # ✅ 区分正确和错误答案（使用adaptive threshold）
    #         rewards_in_group = [original_rewards[i] for i in indices]
    #         max_reward = max(rewards_in_group)
            
    #         # 只有当有高质量答案时才计算R_align
    #         if max_reward < 0.5:
    #             continue
            
    #         # 将reward > 0.7*max_reward的视为"好答案"
    #         correct_threshold = 0.7 * max_reward
    #         correct_indices = [i for i in indices if original_rewards[i] >= correct_threshold]
            
    #         if len(correct_indices) <= 1:
    #             continue
            
    #         groups_processed += 1
    #         group_answers = [answers[i] for i in correct_indices]
            
    #         # 计算相似度矩阵（只在正确答案间）
    #         c3_sim_matrix = np.zeros((len(correct_indices), len(correct_indices)))
    #         for i in range(len(correct_indices)):
    #             for j in range(len(correct_indices)):
    #                 if i == j:
    #                     c3_sim_matrix[i, j] = 1.0
    #                 else:
    #                     c3_sim_matrix[i, j] = self._compute_character_3_similarity(
    #                         group_answers[i], group_answers[j]
    #                     )
            
    #         semantic_sim_matrix = self._compute_semantic_similarity(group_answers)
    #         combined_sim_matrix = (self.c3_weight * c3_sim_matrix + 
    #                             self.semantic_weight * semantic_sim_matrix)
            
    #         # ✅ 关键改进：使用相对一致性（减去该组的平均值）
    #         for local_i, global_i in enumerate(correct_indices):
    #             other_sims = [combined_sim_matrix[local_i, local_j] 
    #                         for local_j in range(len(correct_indices)) if local_j != local_i]
                
    #             if other_sims:
    #                 avg_sim = np.mean(other_sims)
    #                 group_mean_sim = np.mean(combined_sim_matrix[np.triu_indices(len(correct_indices), k=1)])
                    
    #                 # 相对一致性：高于组平均的给正奖励，低于的给负奖励
    #                 relative_align = avg_sim - group_mean_sim
                    
    #                 # ✅ 根据正确答案数量调整权重（样本越多，一致性越重要）
    #                 group_weight = min(1.0, len(correct_indices) / 5.0)
                    
    #                 r_align_scores[global_i] = relative_align * group_weight
        
    #     if groups_processed > 0:
    #         print(f"[R_align_v2] Processed {groups_processed} question groups with multiple correct answers")
    #         valid_scores = r_align_scores[r_align_scores != 0]
    #         if len(valid_scores) > 0:
    #             print(f"[R_align_v2] Mean: {np.mean(valid_scores):.4f}, "
    #                 f"Std: {np.std(valid_scores):.4f}, "
    #                 f"Range: [{np.min(valid_scores):.4f}, {np.max(valid_scores):.4f}]")
        
    #     return r_align_scores


    # def _compute_single_metric_v2(self, data: DataProto, metric_name: str):
    #     """改进的奖励计算"""
    #     reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    #     already_print_data_sources = {}
    #     original_rewards = []
        
    #     # 第一遍：计算原始奖励
    #     for i in range(len(data)):
    #         data_item = data[i]
    #         prompt_ids = data_item.batch['prompts']
    #         prompt_length = prompt_ids.shape[-1]
    #         valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
    #         valid_prompt_ids = prompt_ids[-valid_prompt_length:]
    #         response_ids = data_item.batch['responses']
    #         valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
    #         valid_response_ids = response_ids[:valid_response_length]

    #         sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    #         sequences_str = self.tokenizer.decode(sequences)
    #         ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

    #         data_source = data_item.non_tensor_batch['data_source']
    #         compute_score_fn = self._select_rm_score_fn(data_source, metric_name)
    #         score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
    #                                 format_score=self.format_score)
    #         reward_tensor[i, valid_response_length - 1] = score
    #         original_rewards.append(score)

    #         if data_source not in already_print_data_sources:
    #             already_print_data_sources[data_source] = 0
    #         if already_print_data_sources[data_source] < self.num_examine:
    #             already_print_data_sources[data_source] += 1
    #             print(f"=== {metric_name.upper()} Evaluation ===")
    #             print(f"Score: {score}")
    #             print(sequences_str[:200])
        
    #     # 第二遍：计算并融合R_align
    #     if self.is_training and self.r_align_weight > 0:
    #         r_align_scores = self._compute_r_align(data, np.array(original_rewards))
            
    #         # ✅ 标准化R_align到合理范围 [-0.1, 0.1]
    #         r_align_scores = np.clip(r_align_scores, -0.2, 0.2)
            
    #         for i in range(len(data)):
    #             data_item = data[i]
    #             prompt_ids = data_item.batch['prompts']
    #             prompt_length = prompt_ids.shape[-1]
    #             response_ids = data_item.batch['responses']
    #             # valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    #             valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())

                
    #             # ✅ 加权组合：保持原始奖励的主导地位
    #             combined_reward = original_rewards[i] + self.r_align_weight * r_align_scores[i]
                
    #             # ✅ 只clip下限，允许正确+一致的答案获得>1的奖励
    #             combined_reward = max(0.0, min(combined_reward, self.max_combined_reward))
                
    #             reward_tensor[i, valid_response_length - 1] = combined_reward

    #     return reward_tensor
    def _compute_r_align(self, data: DataProto, original_rewards):
        """
        Anti-consistency 版本的 R_align：
        - 只在“明显错误”的答案之间计算相似度
        - 如果一个错误答案和其他错误答案非常相似，则给一个负向惩罚
        - 正确答案不受影响
        """
        answers = self._extract_and_normalize_answers(data)
        uids = data.non_tensor_batch.get('uid', None)
        n = len(answers)
        r_align_scores = np.zeros(n, dtype=np.float32)
        
        if uids is None:
            return r_align_scores
        
        # 把样本按 uid 分组（同一问题的多条采样）
        uid_to_indices = defaultdict(list)
        for i, uid in enumerate(uids):
            uid_to_indices[uid].append(i)
        
        groups_processed = 0
        bad_threshold = 0.4   # c3recall < 0.4 视为“明显错误”
        margin = 0.5          # 只有相似度 > margin 才真正惩罚
        # bad_threshold = 0.2   # c3recall < 0.4 视为“明显错误”
        # margin = 0.7    
        
        for uid, indices in uid_to_indices.items():
            if len(indices) <= 1:
                continue
            
            # 挑出这一组里“明显错误”的样本
            rewards_in_group = [original_rewards[i] for i in indices]
            bad_indices = [i for i in indices if original_rewards[i] < bad_threshold]
            
            # 没有足够的错误样本就不做 anti-consistency
            if len(bad_indices) <= 1:
                continue
            
            groups_processed += 1
            group_answers = [answers[i] for i in bad_indices]
            
            # 1) 计算错误答案之间的相似度矩阵
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
            
            # semantic_sim_matrix = self._compute_semantic_similarity(group_answers)
            combined_sim_matrix = (self.c3_weight * c3_sim_matrix)
            
            # 2) 对每个错误样本，找到它和其他错误样本中“最像”的那个
            for local_i, global_i in enumerate(bad_indices):
                other_sims = [combined_sim_matrix[local_i, local_j]
                            for local_j in range(k) if local_j != local_i]
                if not other_sims:
                    continue
                
                max_sim = float(np.max(other_sims))
                # 只有当“相似度明显高于 margin”时，才施加惩罚
                # 惩罚值 ∝ max(0, max_sim - margin)
                penalty_raw = max(0.0, max_sim - margin)
                
                if penalty_raw <= 0.0:
                    continue
                
                # 根据错误样本数量调整权重：错误样本多，则适当加大一点惩罚
                group_weight = min(1.0, len(bad_indices) / 5.0)
                
                # 注意是负号：相似度越高，惩罚越大
                r_align_scores[global_i] = - penalty_raw * group_weight
        
        if groups_processed > 0:
            valid_scores = r_align_scores[r_align_scores != 0]
            if len(valid_scores) > 0:
                print(f"[R_anti_align] Processed {groups_processed} question groups with multiple bad answers")
                print(f"[R_anti_align] Mean: {np.mean(valid_scores):.4f}, "
                    f"Std: {np.std(valid_scores):.4f}, "
                    f"Range: [{np.min(valid_scores):.4f}, {np.max(valid_scores):.4f}]")
        
        return r_align_scores
    
    def _compute_single_metric(self, data: DataProto, metric_name: str):
        """改进的奖励计算"""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        original_rewards = []
        
        # 第一遍：计算原始奖励
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
        
        # 第二遍：计算并融合R_align
        if self.is_training and self.r_align_weight > 0:
            r_align_scores = self._compute_r_align(data, np.array(original_rewards))
            
            # 这里确保 r_align_scores 不会太大幅度拉低 reward
            # 只允许适度的负偏移，例如 [-0.5, 0]
            r_align_scores = np.clip(r_align_scores, -0.5, 0.0)
            
            for i in range(len(data)):
                data_item = data[i]
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
                
                combined_reward = original_rewards[i] + self.r_align_weight * r_align_scores[i]
                combined_reward = max(0.0, min(combined_reward, self.max_combined_reward))
                
                reward_tensor[i, valid_response_length - 1] = combined_reward

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
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

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

    from verl.trainer.ppo.ray_trainer_parallel import ResourcePoolManager, Role

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

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # # reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, eval_metrics={'em': True})
    # reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, eval_metrics={'c3recall': True})
    # 训练时使用c3recall + R_align
    # reward_fn = RewardManager(
    #     tokenizer=tokenizer, 
    #     num_examine=0, 
    #     eval_metrics={'c3recall': True},
    #     r_align_weight=0.1,  # lambda值，控制R_align的权重
    #     c3_weight=0.5,  # character-3-recall的权重
    #     semantic_weight=0.5  # 语义相似度的权重
    # )
    # reward_fn = RewardManager(
    #     tokenizer=tokenizer, 
    #     num_examine=0, 
    #     eval_metrics={'c3recall': True},
    #     r_align_weight=0.1,
    #     c3_weight=0.3,
    #     semantic_weight=0.7,
    #     r_align_only_correct=True,  # ✅ 只对正确答案加R_align
    #     max_combined_reward=1.5,  # 允许奖励达到2.0
    # )
    reward_fn = RewardManager(
    tokenizer=tokenizer, 
    num_examine=0, 
    eval_metrics={'c3recall': True},
    r_align_weight=0.02,   # 建议先用一个非常保守的惩罚权重
    c3_weight=1.0,         # 建议先只用 char-3 做 anti-consistency
    semantic_weight=0.0,   # 先关掉语义部分，跑通后再加
    max_combined_reward=1.5,
)

    # Note that we always use function-based RM for validation
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
