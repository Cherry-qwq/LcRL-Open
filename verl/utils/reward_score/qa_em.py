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

try:
    from zhconv import convert
    ZHCONV_AVAILABLE = True
except ImportError:
    ZHCONV_AVAILABLE = False
    print("Warning: zhconv not installed. Traditional Chinese will not be converted to Simplified Chinese.")

def contains_chinese(text):
    """检查文本是否包含中文字符"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False
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

    if ZHCONV_AVAILABLE and contains_chinese(text):
        text = convert(text, 'zh-cn')


    punctuation_to_remove = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~！？。，、；：""''（）【】《》〈〉「」『』'
    for p in punctuation_to_remove:
        text = text.replace(p, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(?<=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])', '', text)
    # print(text)
    return text

def get_all_alias(ground_truth_id):

    return set()

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

def flexible_exact_match_score(prediction, ground_truth, ground_truth_id=None):
    """
    灵活精确匹配评分函数
    """
    ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
    if ground_truth_id and isinstance(ground_truth_id, str):
        ground_truths.update(get_all_alias(ground_truth_id))
 
    is_correct = 0
    for gt in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_gt = normalize_answer(gt)
        if normalized_gt in normalized_prediction or normalized_gt.lower() in normalized_prediction or normalized_gt.capitalize() in normalized_prediction:
            is_correct = 1
            break
        
    return is_correct

def character_3gram_recall(prediction, ground_truth, ground_truth_id=None):
    """
    字符3-gram召回率评分函数
    """
    recall = 0
    ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
    if ground_truth_id and isinstance(ground_truth_id, str):
        ground_truths.update(get_all_alias(ground_truth_id))
 
    # 将文本转化为字符3-gram
    def get_3gram(text):
        if len(text) < 3:
            return []
        return [text[i:i+3] for i in range(len(text)-2)]
    
    for gt in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_gt = normalize_answer(gt)
 
        pred_3grams = set(get_3gram(normalized_prediction))
        true_3grams = set(get_3gram(normalized_gt))
        recall_new = len(pred_3grams & true_3grams) / len(true_3grams) if len(true_3grams) > 0 else 0.0
        recall = max(recall_new, recall)
 
    return recall
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

# def extract_solution(solution_str):
#     """Extract the answer from the solution string."""
#     answer_pattern = r'<answer>(.*?)</answer>'
#     matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    
#     if not matches:
#         return None
    
#     # 返回最后一个匹配
#     return matches[-1].strip()

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    filtered_str = re.sub(r'<information>.*?</information>', '', solution_str, flags=re.DOTALL)
    # match <answer>...</answer>
    answer_pattern1 = r'<answer>(.*?)</answer>'
    # match <answer_language=xx> ... </answer_language=xx>
    # answer_pattern2 = r'<answer language="(\w+)">(.*?)</answer>'
    answer_pattern2 = r'<answer\s+language="[^"]*">\s*(.*?)\s*</answer>'
    
    matches1 = re.findall(answer_pattern1, solution_str, re.DOTALL)
    matches2 = re.findall(answer_pattern2, solution_str, re.DOTALL)
    
    # 合并所有匹配
    all_matches = matches1 + matches2
    
    if not all_matches:
        return " "
    
    # 返回最后一个匹配
    return all_matches[-1].strip()
# def extract_solution(solution_str):
#     """
#     针对 RAG + RL 场景优化，增加预清洗步骤
#     """
#     if solution_str is None:
#         return None

#     # 1. 预清洗：定义那句固定的纠错文本
#     # 注意：建议尽量匹配完整句子，防止误删其他内容
#     error_msg_pattern = (
#         r"My previous action is invalid\. If I want to search, I should put the query between "
#         r"<search> and </search>\. If I want to give the final answer, I should put the answer between "
#         r"<answer> and </answer>\. Let me try again\.?"
#     )
    
#     # 将其替换为空字符串 (flags=re.IGNORECASE 防止大小写变动导致匹配失败)
#     content_cleaned = re.sub(error_msg_pattern, "", solution_str, flags=re.IGNORECASE)

#     # 2. 分割 Prompt 和 Generation (依然保留这个逻辑作为双重保险)
#     if "<|im_start|>assistant" in content_cleaned:
#         content_to_parse = content_cleaned.split("<|im_start|>assistant")[-1]
#     else:
#         content_to_parse = content_cleaned

#     # 3. 正则提取答案 (使用之前的优化版正则)
#     # 允许不闭合，兼容带属性的标签
#     answer_pattern = r'<answer[^>]*>(.*?)(?:</answer>|$)'
#     matches = re.findall(answer_pattern, content_to_parse, re.DOTALL | re.IGNORECASE)
    
#     if not matches:
#         return None
    
#     final_answer = matches[-1].strip()

#     # 4. 反作弊逻辑 (依然保留)
#     # 此时因为 error_msg 已经被删掉了，所以不会因为 error_msg 里包含 <search> 而误杀
#     forbidden_tags = ["<think>", "</think>", "<information>", "</information>"]
#     for tag in forbidden_tags:
#         if tag in final_answer:
#             return None

#     # 5. 长度熔断
#     if len(final_answer) > 2000: # 根据多语言任务适当放宽一点
#         return None

#     return final_answer

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

def compute_score_fem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for flexible exact match - 用于验证时的额外评估"""
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        ground_truth_id = ground_truth.get('target_id', None)
        return flexible_exact_match_score(answer, ground_truth['target'], ground_truth_id)

def compute_score_c3recall(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for character 3-gram recall - 用于验证时的额外评估"""
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        ground_truth_id = ground_truth.get('target_id', None)
        return character_3gram_recall(answer, ground_truth['target'], ground_truth_id)

# # Copyright 2024 Bytedance Ltd. and/or its affiliates

# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at

# #     http://www.apache.org/licenses/LICENSE-2.0

# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import re
# import string
# import random
# import unicodedata

# try:
#     from zhconv import convert
#     ZHCONV_AVAILABLE = True
# except ImportError:
#     ZHCONV_AVAILABLE = False
#     print("Warning: zhconv not installed. Traditional Chinese will not be converted to Simplified Chinese.")

# def contains_chinese(text):
#     """检查文本是否包含中文字符"""
#     for char in text:
#         if '\u4e00' <= char <= '\u9fff':
#             return True
#     return False

# def normalize_answer(s):

#     if s is None:
#         return "None"

#     text = s.lower()
#     text = unicodedata.normalize('NFKC', text)

#     if ZHCONV_AVAILABLE and contains_chinese(text):
#         text = convert(text, 'zh-cn')


#     punctuation_to_remove = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~！？。，、；：""''（）【】《》〈〉「」『』'
#     for p in punctuation_to_remove:
#         text = text.replace(p, ' ')
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = re.sub(r'(?<=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])', '', text)
#     # print(text)
#     return text

# def get_all_alias(ground_truth_id):

#     return set()

# def em_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer == normalized_prediction:
#             score = 1
#             break
#     return score


# def subem_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer in normalized_prediction:
#             score = 1
#             break
#     return score

# def flexible_exact_match_score(prediction, ground_truth, ground_truth_id=None):
#     """
#     灵活精确匹配评分函数
#     """
#     ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
#     if ground_truth_id and isinstance(ground_truth_id, str):
#         ground_truths.update(get_all_alias(ground_truth_id))
 
#     is_correct = 0
#     for gt in ground_truths:
#         normalized_prediction = normalize_answer(prediction)
#         normalized_gt = normalize_answer(gt)
#         if normalized_gt in normalized_prediction or normalized_gt.lower() in normalized_prediction or normalized_gt.capitalize() in normalized_prediction:
#             is_correct = 1
#             break
        
#     return is_correct

# def character_3gram_recall(prediction, ground_truth, ground_truth_id=None):
#     """
#     字符3-gram召回率评分函数
#     """
#     recall = 0
#     ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
#     if ground_truth_id and isinstance(ground_truth_id, str):
#         ground_truths.update(get_all_alias(ground_truth_id))
 
#     # 将文本转化为字符3-gram
#     def get_3gram(text):
#         if len(text) < 3:
#             return []
#         return [text[i:i+3] for i in range(len(text)-2)]
    
#     for gt in ground_truths:
#         normalized_prediction = normalize_answer(prediction)
#         normalized_gt = normalize_answer(gt)
 
#         pred_3grams = set(get_3gram(normalized_prediction))
#         true_3grams = set(get_3gram(normalized_gt))
#         recall_new = len(pred_3grams & true_3grams) / len(true_3grams) if len(true_3grams) > 0 else 0.0
#         recall = max(recall_new, recall)
 
#     return recall

# def extract_solution(solution_str):
#     """Extract the answer from the solution string."""
#     target_token = "<|im_start|>assistant"
#     if target_token in solution_str:
#         solution_str = solution_str[solution_str.rfind(target_token):]
#     solution_str = solution_str + "</answer>"
#     # match <answer>...</answer>
#     answer_pattern1 = r'<answer>(.*?)</answer>'
#     # match <answer_language=xx> ... </answer_language=xx>
#     answer_pattern2 = r'<answer\s+language="[^"]*">\s*(.*?)\s*</answer>'
    
#     matches1 = re.findall(answer_pattern1, solution_str, re.DOTALL)
#     matches2 = re.findall(answer_pattern2, solution_str, re.DOTALL)
    
#     # 合并所有匹配
#     all_matches = matches1 + matches2
    
#     if not all_matches:
#         return None
    
#     # 取最后一个匹配作为初步候选
#     final_answer = all_matches[-1].strip()

#     # =========================================================
#     # 修改区域：清洗与过滤逻辑
#     # =========================================================

#     # 1. 黑名单过滤：去除 <information> ... </information>
#     # 使用正则替换，re.DOTALL 确保能匹配跨行内容
#     invalid_action_msg = "<answer> and </answer>"
#     if invalid_action_msg in final_answer:
#         final_answer = final_answer.replace(invalid_action_msg, "")
#     if '<information>' in final_answer:
#         final_answer = re.sub(r'<information>.*?</information>', '', final_answer, flags=re.DOTALL)

#     # 2. 递归清洗：处理嵌套/多余的 <answer> 标签
#     # 查找内容中是否还遗留有 <answer> 或 <answer language="...">
#     # 如果有，我们只取最后一个标签之后的内容
#     inner_pattern = r'<answer(?: +language="[^"]*")?>'
#     inner_matches = list(re.finditer(inner_pattern, final_answer))
    
#     if inner_matches:
#         # 找到最后一个匹配的结束位置
#         last_match = inner_matches[-1]
#         # 截取该位置之后的所有内容
#         final_answer = final_answer[last_match.end():]

#     # 再次去除首尾空白
#     final_answer = final_answer.strip()

#     # 防止清洗后变为空字符串
#     if not final_answer:
#         return None

#     # =========================================================
    
#     return final_answer

# def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
#     """The scoring function for exact match (EM).

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
#     answer = extract_solution(solution_str=solution_str)
#     do_print = random.randint(1, 64) == 1
    
#     if do_print:
#         print(f"--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")
    
#     if answer is None:
#         return 0
#     else:
#         if em_check(answer, ground_truth['target']):
#             return score
#         else:
#             return format_score


# def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
#     """The scoring function for substring exact match (EM).

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
#     answer = extract_solution(solution_str=solution_str)
#     do_print = random.randint(1, 64) == 1
    
#     if do_print:
#         print(f"--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")
    
#     if answer is None:
#         return 0
#     else:
#         if subem_check(answer, ground_truth['target']):
#             return score
#         else:
#             return format_score

# def compute_score_fem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
#     """The scoring function for flexible exact match - 用于验证时的额外评估"""
#     answer = extract_solution(solution_str=solution_str)
#     if answer is None:
#         return 0
#     else:
#         ground_truth_id = ground_truth.get('target_id', None)
#         return flexible_exact_match_score(answer, ground_truth['target'], ground_truth_id)

# def compute_score_c3recall(solution_str, ground_truth, method='strict', format_score=0., score=1.):
#     """The scoring function for character 3-gram recall - 用于验证时的额外评估"""
#     answer = extract_solution(solution_str=solution_str)
#     if answer is None:
#         return 0
#     else:
#         ground_truth_id = ground_truth.get('target_id', None)
#         return character_3gram_recall(answer, ground_truth['target'], ground_truth_id)