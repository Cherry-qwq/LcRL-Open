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
Preprocess the MKQA dataset to parquet format with multi-language support
"""

import pandas as pd
import os
import argparse

def filter_null_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out samples containing null ground truth.
    Criteria: If the text of any answer in any language is None, the row is considered to have a null answer.
    """

    def has_null_answer(answers):
        
        if not isinstance(answers, dict):
            return True  # Treat structural errors as invalid samples

        for lang, answer_list in answers.items():
            if answer_list is None:
                return True
            for answer in answer_list:
                if answer.get("text") is None:
                    return True
        return False

    mask = ~df["answers"].apply(has_null_answer)
    kept = mask.sum()
    removed = len(df) - kept
    print(f"[Filter] Total rows: {len(df)}, kept: {kept}, removed (null ground truth): {removed}")
    return df[mask].reset_index(drop=True)

LANGUAGE_TO_SOURCE = {
        'fr': 'it',  
        'es': 'fr',   
        'pt': 'fr',  
        'it': 'fr',   
        'de': 'fr',   
        'en': 'fr',   
        'ru': 'fr',  
        'zh_cn': 'ja',
        'zh': 'ja',
        'ja': 'zh_cn',   
        'ko': 'ja',   
        'th': 'ja',   
        'fi': 'ru',  
        'ar': 'fr', 
    }

def make_prefix(dp, lang, template_type='base', question_lang=None):
    """
    Generate multilingual prompt, supporting multiple template types.
    """
    if question_lang is None:
        question_lang = lang
    question = dp.queries[question_lang]
    question = question.strip()
    
    # Ensure the question ends with a question mark
    if question[-1] not in ['?', '？', '؟']:
        question += '?'
    
    # Language name mapping for prompts
    language_names = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'zh_cn': 'Chinese',
        'ja': 'Japanese',
        'ar': 'Arabic',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'it': 'Italian',
        'ko': 'Korean',
        'fi': 'Finnish',
        'th': 'Thai',
    }
    
    language_to_high_resource = {
        # Romance languages
        'fr': 'Italian',   
        'es': 'French',   
        'pt': 'French',   
        'it': 'French',   
        'de': 'French',   
        'en': 'French',   
        'ru': 'French',   
        
        # Sino-Tibetan / East Asian
        'zh_cn': 'Japanese',
        'ja': 'Chinese',
        'ko': 'Japanese',   
        'th': 'Japanese',   
        
        'fi': 'Russian',   
        'ar': 'French',  
    }
    

    instr_lang = question_lang
    lang_name = language_names.get(instr_lang, 'English')
    high_resource_lang = language_to_high_resource.get(instr_lang, 'English')
    
    # Select example answers based on language
    example_answers = {
        'en': 'Beijing',
        'fr': 'l’arctique',
        'es': 'El ártico',
        'de': 'arktis',
        'zh_cn': '北京',
        'ja': '東京',
        'ar': 'القاهرة',
        'ru': 'Москва',
        'pt': 'O pólo norte',
        'it': 'artico',
        'ko': '서울',
        'hi': 'दिल्ली',
        'tr': 'Ankara',
    }

    answer_lang_name = language_names.get(lang, 'English')
    example_answer = example_answers.get(lang, 'Answer')
    
    if template_type == 'base':
        # Base template for all languages (English instructions)
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. \
Note: Answers should be in {answer_lang_name}. For example, <answer> {example_answer} </answer>. 
Question: {question}"""
        
    elif template_type == 'multiingual':
        # Multilingual template placeholder
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> and </search>, and it will return the top searched results between <information> and </information>. \
You can search at least 2 times: first search in {lang_name} knowledge base, and second search in English and other languages like {high_resource_lang} to provide comprehensive information and resolve potential conflicts. If needed, continue searching in English \
If no further external knowledge is needed, provide the answer inside <answer> and </answer> without detailed illustrations. \
Note: The answer must be in {answer_lang_name}. For example, <answer> {example_answer} </answer>. 

Question: {question}"""
    
    else:
        raise NotImplementedError(f"Template type '{template_type}' not implemented")
    
    return prefix


def create_dataset(df, language, split, template_type='base'):
    data = []
    
    for idx, row in enumerate(df.itertuples(), start=0):
        target_answers = []
        for answer_item in row.answers[language]:
            if 'text' in answer_item:
                target_answers.append(answer_item['text'])
            if 'aliases' in answer_item:
                target_answers.extend(answer_item['aliases'])
        question_prompt = make_prefix(row, language, template_type=template_type)

        # Construct content_en and content_source
        content_en = None
        if isinstance(row.queries, dict) and 'en' in row.queries and row.queries['en'] is not None:
            try:
                # Reuse the same template logic with lang='en'
                content_en = make_prefix(row, lang=language, template_type=template_type, question_lang='en' )
            except Exception as e:
                print(f"[WARN] Failed to build content_en for idx={idx}, language={language}: {e}")
                content_en = None
        
        content_source = None
        source_lang = LANGUAGE_TO_SOURCE.get(language)
        if source_lang is not None:
            if isinstance(row.queries, dict) and source_lang in row.queries and row.queries[source_lang] is not None:
                try:
                    content_source = make_prefix(row, lang=language, template_type=template_type, question_lang=source_lang)
                except Exception as e:
                    print(f"[WARN] Failed to build content_source for idx={idx}, language={language}, source_lang={source_lang}: {e}")
                    content_source = None
            else:
                print(f"[INFO] No query found for source_lang={source_lang} when processing language={language}, idx={idx}")

        solution = {
            "target": target_answers,
        }
        
        example = {
            "id": f"{language}_{split}_{idx}",
            "question": row.queries[language].strip(),
            "golden_answers": row.answers[language],
            "language": language,  
            "data_source": "mkqa",
            "prompt": [{
                "role": "user",
                "content": question_prompt,
            }],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'language': language,  
            },
            "example_id":row.example_id
        }
        
        data.append(example)
    
    return data


def process_language(train_path, test_path, language, template_type, local_dir, hdfs_dir=None):
    """
    Process data for a single language:
    1. Read Train and Test jsonl files separately.
    2. Filter out samples with null ground truth.
    3. Convert to Parquet format and save.
    """
    print(f"\n{'='*60}")
    print(f"Processing language: {language.upper()}")
    print(f"{'='*60}")
    
    # --- Process Train Set ---
    print(f"Loading TRAIN set from: {train_path}")
    if os.path.exists(train_path):
        train_df = pd.read_json(train_path, lines=True)
        print(f"[Train Raw] Total rows: {len(train_df)}")
        train_df = filter_null_answers(train_df)
        print(f"[Train Filtered] Rows kept: {len(train_df)}")
        train_data = create_dataset(train_df, language, 'train', template_type)
    else:
        print(f"[ERROR] Train file not found: {train_path}")
        return

    # --- Process Test Set ---
    print(f"Loading TEST set from: {test_path}")
    if os.path.exists(test_path):
        test_df = pd.read_json(test_path, lines=True)
        print(f"[Test Raw] Total rows: {len(test_df)}")
        test_df = filter_null_answers(test_df)
        print(f"[Test Filtered] Rows kept: {len(test_df)}")
        test_data = create_dataset(test_df, language, 'test', template_type)
    else:
        print(f"[ERROR] Test file not found: {test_path}")
        return

    # --- Save ---
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    train_file = os.path.join(local_dir, f'train_{language}.parquet')
    test_file = os.path.join(local_dir, f'test_{language}.parquet')
    
    pd.DataFrame(train_data).to_parquet(train_file)
    pd.DataFrame(test_data).to_parquet(test_file)
    
    print(f"✓ Saved Train: {train_file} (Count: {len(train_data)})")
    print(f"✓ Saved Test:  {test_file}  (Count: {len(test_data)})")
    
    # HDFS storage (if needed)
    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)
        copy(src=train_file, dst=os.path.join(hdfs_dir, f'train_{language}.parquet'))
        copy(src=test_file, dst=os.path.join(hdfs_dir, f'test_{language}.parquet'))
        print(f"✓ Uploaded to HDFS: {hdfs_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/mkqa/cross', help='Local directory to save parquet files')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--template_type', type=str, default='base', 
                       choices=['base', 'multilang_instruction', 'base-hr'],
                       help='Template type for prompt generation')
    parser.add_argument('--languages', type=str, default='all',
                       help='Comma-separated language codes (e.g., "en,fr,es") or "all" for all languages')
    
    # Receive train and test file paths separately
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to the existing TRAIN jsonl file')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to the existing TEST jsonl file')

    args = parser.parse_args()
    
    # Define supported languages
    all_languages = ['en', 'fr', 'es', 'de', 'zh_cn', 'ja', 'ar', 'ru', 'pt', 'it', 'ko', 'fi', 'th']
    
    # Parse languages to process
    if args.languages == 'all':
        languages_to_process = all_languages
    else:
        languages_to_process = [lang.strip() for lang in args.languages.split(',')]
        invalid_langs = [lang for lang in languages_to_process if lang not in all_languages]
        if invalid_langs:
            print(f"[WARNING] Invalid language codes: {invalid_langs}")
            print(f"Supported languages: {all_languages}")
            languages_to_process = [lang for lang in languages_to_process if lang in all_languages]
    
    if not languages_to_process:
        print("[ERROR] No valid languages to process!")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"Will process {len(languages_to_process)} languages:")
    print(f"{', '.join(languages_to_process)}")
    print(f"Template type: {args.template_type}")
    print(f"Train File: {args.train_file}")
    print(f"Test File: {args.test_file}")
    print(f"{'='*60}")
    
    # Process each language
    for language in languages_to_process:
        try:
            process_language(
                train_path=args.train_file,
                test_path=args.test_file,
                language=language,
                template_type=args.template_type,
                local_dir=args.local_dir,
                hdfs_dir=args.hdfs_dir
            )
        except Exception as e:
            print(f"[ERROR] Failed to process language {language}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("✓ All languages processed successfully!")
    print(f"Output directory: {args.local_dir}")
    print(f"{'='*60}\n")