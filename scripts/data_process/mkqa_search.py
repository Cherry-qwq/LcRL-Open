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
Preprocess the nq dataset to parquet format
"""

# import re
# import os
# import datasets
# import pandas as pd
# from verl.utils.hdfs_io import copy, makedirs
# import argparse
import pandas as pd
import os
import datasets
from sklearn.model_selection import train_test_split
import json
import argparse
# from langdetect import detect
from sklearn.model_selection import train_test_split

def make_prefix(dp, lang, template_type):
    # question = dp['queries'][lang]
    question = dp.queries[lang]
    question = question.strip()
    if question[-1] not in ['?', '？']:
            question += '?'

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 'special':
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Note: Answers should be in French. For example, <answer> Tartes aux œufs </answer>. 
Question: {question}\n"""
    elif template_type == 'label':
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer language="French"> and </answer>, without detailed illustrations. Note: Answers should be in French. For example, <answer language="French"> Tartes aux œufs </answer>. 
        
        """
    elif template_type == 'zh_cn':
        prefix = f"""请回答以下问题。每次获得新信息后，你必须在<think>和</think>之间进行推理。推理后，如果发现缺少某些知识，可以通过<search> 问题 </search>调用搜索引擎，搜索结果将在<information>和</information>之间返回。你可以根据需要多次搜索。如果不需要进一步的外部知识，可以直接在<answer>和</answer>之间提供答案。答案直接给出，无需进行详细解释，例如：<answer> 北京 </answer>。问题：{question}"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/mkqa')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--language', type=str, default='en')


    args = parser.parse_args()

    file_path = '/local_data/ruiqi/Search-R1/data/mkqa/mkqa.jsonl'
    df = pd.read_json(file_path, lines=True)

    # 划分训练集和测试集
    test_df, train_df = train_test_split(df, test_size=0.8, random_state=42, shuffle=False)

    language = args.language
    type = args.template_type
    # add a row to each data item that represents a unique id
    def create_dataset(df, language, split):
        data = []
        # for idx, row in df.iterrows():
        for idx, row in enumerate(df.itertuples(), start=0):
            example = {
            "id": f"{split}_{idx}",
            # "question": row['queries'][language].strip(),
            "question": row.queries[language].strip(),
            # "golden_answers": row['answers'][language]
            "golden_answers": row.answers[language]
        }



            question = make_prefix(row, language, template_type=args.template_type)
            # solution = {
            #     # "target": row['answers'][language],  # 选择英语作为标准答案
            #     "target": row.answers[language],
            # }
            target_answers = []
            for answer_item in row.answers[language]:
                # 添加text
                if 'text' in answer_item:
                    target_answers.append(answer_item['text'])
                # 添加所有aliases
                if 'aliases' in answer_item:
                    target_answers.extend(answer_item['aliases'])

            solution = {
                "target": target_answers,
            }

            example.update(  {
                "data_source": "mkqa",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            })
            data.append(example)
        return data
    
  
    train_data = create_dataset(train_df, language, 'train')
    test_data = create_dataset(test_df, language, 'test')

    # 保存为 Parquet 格式
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # 保存数据到文件
    if type == "special":
        train_df_parquet = os.path.join(local_dir, f'train_{language}.parquet')
        test_df_parquet = os.path.join(local_dir, f'test_{language}.parquet')
    else:
        train_df_parquet = os.path.join(local_dir, f'train_{language}_{type}.parquet')
        test_df_parquet = os.path.join(local_dir, f'test_{language}_{type}.parquet')

    # 使用 Pandas DataFrame 存储并保存
    pd.DataFrame(train_data).to_parquet(train_df_parquet)
    pd.DataFrame(test_data).to_parquet(test_df_parquet)

    if hdfs_dir is not None:
        # HDFS 存储
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print("数据集处理完成，保存至 Parquet 文件。")
