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

import re
import string
import random
import unicodedata
from typing import Union, List
# def normalize_answer(s):
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         if text is None:
#             none_list= ["None"]
#             return none_list
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_answer(s):

    if s is None:
        return "None"

    text = s.lower()
    text = unicodedata.normalize('NFKC', text)
    punctuation_to_remove = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~！？。，、；：""''（）【】《》〈〉「」『』'
    for p in punctuation_to_remove:
        text = text.replace(p, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(?<=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])', '', text)
    
    return text


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


# def extract_solution(solution_str):
#     """Extract the equation from the solution string."""
#     # Remove everything before the first "Assistant:"
#     # if "Assistant:" in solution_str:
#     #     solution_str = solution_str.split("Assistant:", 1)[1]
#     # elif "<|im_start|>assistant" in solution_str:
#     #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
#     # else:
#     #     return None
#     # solution_str = solution_str.split('\n')[-1]

#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.finditer(answer_pattern, solution_str, re.DOTALL)
#     matches = list(match)
    
#     # If there are 0 or exactly 1 matches, return None
#     if len(matches) <= 1:
#         return None
    
#     # If there are 2 or more matches, return the last one
#     return matches[-1].group(1).strip()

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    
    if not matches:
        return None
    
    # 返回最后一个匹配
    return matches[-1].strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
class EvaluationMetrics:
    """评测指标工具类"""
    
    @staticmethod
    def get_all_alias(ground_truth_id):
        """获取所有别名，目前返回空集合，可以根据需要扩展"""
        # TODO: 实现根据ground_truth_id获取别名的逻辑
        return set()
    
    @classmethod
    def flexible_exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
 
        is_correct = 0
        for gt in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_gt = normalize_answer(gt)
            if normalized_gt in normalized_prediction or normalized_gt.lower() in normalized_prediction or normalized_gt.capitalize() in normalized_prediction:
                is_correct = 1
                break
        
        return is_correct
    
    @classmethod
    def character_3gram_recall(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        recall = 0
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
 
        # 将文本转化为字符3-gram
        def get_3gram(text):
            return [text[i:i+3] for i in range(len(text)-2)]
        
        for gt in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_gt = normalize_answer(gt)
 
            pred_3grams = set(get_3gram(normalized_prediction))
            true_3grams = set(get_3gram(normalized_gt))
            recall_new = len(pred_3grams & true_3grams) / len(true_3grams) if len(true_3grams) > 0 else 0.0
            recall = max(recall_new, recall)
 
        return recall



# new_added
def compute_multiple_scores(solution_str, ground_truth, ground_truth_id=None):

    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return {
            'em': 0.0,
            'fem': 0.0,
            'c_3_recall': 0.0
        }
    
    target = ground_truth['target']
    
    em_score = em_check(answer, target)

    fem_score = EvaluationMetrics.flexible_exact_match_score(
        prediction=answer,
        ground_truth=target,
        ground_truth_id=ground_truth_id
    )
    
    c_3_recall_score = EvaluationMetrics.character_3gram_recall(
        prediction=answer,
        ground_truth=target,
        ground_truth_id=ground_truth_id
    )
    
    return {
        'em': float(em_score),
        'fem': float(fem_score),
        'c_3_recall': float(c_3_recall_score)
    }