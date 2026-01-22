# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

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
    """Check if the text contains Chinese characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def normalize_answer(s):
    """Normalize answer string by converting to lowercase, NFKC normalization, and punctuation removal."""
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
    # Remove spaces between CJK characters
    text = re.sub(r'(?<=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])', '', text)
    return text

def get_all_alias(ground_truth_id):
    """Retrieve aliases based on ground truth ID."""
    return set()

def em_check(prediction, golden_answers):
    """Exact Match check."""
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
    """Substring Exact Match check."""
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
    """Flexible Exact Match scoring function."""
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
    """Character 3-gram recall scoring function."""
    recall = 0
    ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
    if ground_truth_id and isinstance(ground_truth_id, str):
        ground_truths.update(get_all_alias(ground_truth_id))
 
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

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    solution_str = solution_str + '</answer>'
    error_msg_pattern = (
        r"My previous action is invalid\. If I want to search, I should put the query between "
        r"<search> and </search>\. If I want to give the final answer, I should put the answer between "
        r"<answer> and </answer>\. Let me try again\.?"
    )
    solution_str = re.sub(error_msg_pattern, "", solution_str, flags=re.DOTALL | re.IGNORECASE)
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant")[-1]
    filtered_str = re.sub(r'<information>.*?</information>', '', solution_str, flags=re.DOTALL)
    
    
    # match <answer>...</answer>
    answer_pattern1 = r'<answer>(.*?)</answer>'
    answer_pattern2 = r'<answer\s+language="[^"]*">\s*(.*?)\s*</answer>'
    
    # matches1 = re.findall(answer_pattern1, solution_str, re.DOTALL)
    # matches2 = re.findall(answer_pattern2, solution_str, re.DOTALL)
    matches1 = re.findall(answer_pattern1, filtered_str, re.DOTALL)  
    matches2 = re.findall(answer_pattern2, filtered_str, re.DOTALL)

    all_matches = matches1 + matches2
    
    if not all_matches:
        return " "

    return all_matches[-1].strip()

def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """Scoring function for exact match (EM)."""
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """Scoring function for substring exact match."""
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

def compute_score_fem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """Scoring function for flexible exact match for validation evaluation."""
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        ground_truth_id = ground_truth.get('target_id', None)
        return flexible_exact_match_score(answer, ground_truth['target'], ground_truth_id)

def compute_score_c3recall(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """Scoring function for character 3-gram recall for validation evaluation."""
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        ground_truth_id = ground_truth.get('target_id', None)
        return character_3gram_recall(answer, ground_truth['target'], ground_truth_id)