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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._download()
        self._read_files_and_tokenize()
        print("=== Dataset Sample ===")
        sample = self.dataframe.iloc[0]
        print(f"Keys: {sample.keys()}")
        print(f"content: {sample.get('content', 'MISSING')[:100]}")
        print(f"content_en: {sample.get('content_en', 'MISSING')[:100]}")
        print(f"content_source: {sample.get('content_source', 'MISSING')[:100]}")

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    # def __getitem__(self, item):
    #     """
    #     Note that we also return the raw_input_ids so that it can be combined with other chat template
    #     """
    #     row_dict = self.dataframe.iloc[item].to_dict()

    #     chat = row_dict.pop(self.prompt_key)
    #     # ===== 新增 =====
    #     # row_dict['content'] = row_dict.get('content', None)
    #     # row_dict['content_en'] = row_dict.get('content_en', None)
    #     # row_dict['content_source'] = row_dict.get('content_source', None)
        
    #     # # 如果不存在，尝试从 chat 第一条消息中提取（兼容旧数据）
    #     # if row_dict['content'] is None and isinstance(chat, list) and len(chat) > 0:
    #     #     row_dict['content'] = chat[0].get('content', None)
    #     if isinstance(chat, list) and len(chat) > 0 and isinstance(chat[0], dict):
    #         first_message = chat[0]
    #         # 只在 row_dict 中没有时才从 chat 中提取
    #         if 'content' not in row_dict:
    #             row_dict['content'] = first_message.get('content', None)
    #         if 'content_en' not in row_dict:
    #             row_dict['content_en'] = first_message.get('content_en', None)
    #         if 'content_source' not in row_dict:
    #             row_dict['content_source'] = first_message.get('content_source', None)
    #     # ================

    #     if self.tokenizer.chat_template:
    #         prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    #     else:
    #         prompt_with_chat_template = chat[0]['content']
    #     # prompt_with_chat_template = chat

    #     input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
    #                                                                      tokenizer=self.tokenizer,
    #                                                                      max_length=self.max_prompt_length,
    #                                                                      pad_token_id=self.tokenizer.pad_token_id,
    #                                                                      left_pad=True,
    #                                                                      truncation=self.truncation)

    #     position_ids = compute_position_id_with_mask(attention_mask)

    #     row_dict['input_ids'] = input_ids[0]
    #     row_dict['attention_mask'] = attention_mask[0]
    #     row_dict['position_ids'] = position_ids[0]

    #     # encode prompts without chat template
    #     if self.return_raw_chat:
    #         row_dict['raw_prompt'] = chat.tolist()

    #     # add index for each prompt
    #     index = row_dict.get("extra_info", {}).get("index", 0)
    #     row_dict["index"] = index

    #     return row_dict
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)  # prompt_key = 'prompt'
        
        # ===== 修复：从 prompt 的第一条消息中提取三个版本 =====
        if isinstance(chat, list) and len(chat) > 0 and isinstance(chat[0], dict):
            first_message = chat[0]
            # 提取三个版本的 content
            row_dict['content'] = first_message.get('content', '')
            row_dict['content_en'] = first_message.get('content_en', first_message.get('content', ''))
            row_dict['content_source'] = first_message.get('content_source', first_message.get('content', ''))
            
            # 用于 tokenization 的标准 chat 格式（只保留 role 和 content）
            standard_chat = [{'role': first_message.get('role', 'user'), 'content': first_message.get('content', '')}]
        else:
            # 降级方案：如果不是预期格式
            content = str(chat) if not isinstance(chat, list) else chat[0].get('content', str(chat))
            row_dict['content'] = content
            row_dict['content_en'] = content
            row_dict['content_source'] = content
            standard_chat = [{'role': 'user', 'content': content}]
        # ================

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                standard_chat,  # 使用标准格式，避免 tokenizer 看到多余字段
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            prompt_with_chat_template = standard_chat[0]['content']

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = standard_chat  # 保存标准格式

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", item)
        row_dict["index"] = index

        return row_dict

# class RLHFDataset(Dataset):
#     """
#     We assume the dataset contains a column that contains prompts and other information
#     """

#     def __init__(self,
#                  parquet_files: Union[str, List[str]],
#                  tokenizer: PreTrainedTokenizer,
#                  prompt_key='prompt',
#                  max_prompt_length=1024,
#                  filter_prompts=True,
#                  cache_dir='~/.cache/verl/rlhf',
#                  chat_template_func=None,
#                  return_raw_chat=False,
#                  truncation='error'):
#         if not isinstance(parquet_files, (List, ListConfig)):
#             parquet_files = [parquet_files]

#         self.parquet_files = parquet_files
#         self.cache_dir = os.path.expanduser(cache_dir)
#         self.tokenizer = tokenizer

#         self.prompt_key = prompt_key
#         self.max_prompt_length = max_prompt_length
#         self.filter_prompts = filter_prompts

#         self.return_raw_chat = return_raw_chat
#         self.chat_template_func = chat_template_func
#         self.truncation = truncation

#         self._download()
#         self._read_files_and_tokenize()

#     def _download(self):
#         from verl.utils.fs import copy_local_path_from_hdfs
#         for i, parquet_file in enumerate(self.parquet_files):
#             self.parquet_files[i] = copy_local_path_from_hdfs(
#                 src=parquet_file,
#                 cache_dir=self.cache_dir
#             )

#     def _read_files_and_tokenize(self):
#         dataframes = []
#         for parquet_file in self.parquet_files:
#             # read parquet files and cache
#             dataframe = pd.read_parquet(parquet_file)
#             dataframes.append(dataframe)
#         self.dataframe = pd.concat(dataframes)

#         print(f'original dataset len: {len(self.dataframe)}')

#         # nvm if prompt is too long
#         # tokenizer = self.tokenizer
#         # prompt_key = self.prompt_key
#         # self.dataframe = self.dataframe[self.dataframe.apply(
#         #     lambda doc: len(
#         #         tokenizer.apply_chat_template(
#         #             doc[prompt_key],
#         #             add_generation_prompt=True
#         #         )
#         #     ) <= self.max_prompt_length,
#         #     axis=1
#         # )]

#         print(f'filter dataset len: {len(self.dataframe)}')

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, item):
#         """
#         Note that we also return the raw_input_ids so that it can be combined with other chat template
#         """
#         row_dict = self.dataframe.iloc[item].to_dict()

#         # 这里保持和原版一致：prompt / chat 是一个“对话列表”
#         chat = row_dict.pop(self.prompt_key)

#         if self.tokenizer.chat_template:
#             # chat 必须是 list-like of dicts，而不是 str
#             prompt_with_chat_template = self.tokenizer.apply_chat_template(
#                 chat,
#                 add_generation_prompt=True,
#                 tokenize=False
#             )
#         else:
#             # 没有 chat_template 时直接用第一条 content
#             prompt_with_chat_template = chat[0]['content']

#         input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
#             prompt=prompt_with_chat_template,
#             tokenizer=self.tokenizer,
#             max_length=self.max_prompt_length,
#             pad_token_id=self.tokenizer.pad_token_id,
#             left_pad=True,
#             truncation=self.truncation
#         )

#         position_ids = compute_position_id_with_mask(attention_mask)

#         row_dict['input_ids'] = input_ids[0]
#         row_dict['attention_mask'] = attention_mask[0]
#         row_dict['position_ids'] = position_ids[0]

#         # encode prompts without chat template
#         if self.return_raw_chat:
#             # 如果 chat 是 ndarray，就 tolist；否则直接用
#             if hasattr(chat, "tolist"):
#                 row_dict['raw_prompt'] = chat.tolist()
#             else:
#                 row_dict['raw_prompt'] = chat

#         # add index for each prompt
#         index = row_dict.get("extra_info", {}).get("index", 0)
#         row_dict["index"] = index

#         return row_dict
# class RLHFDataset(Dataset):
#     """
#     We assume the dataset contains a column that contains prompts and other information
#     """

#     def __init__(self,
#                  parquet_files: Union[str, List[str]],
#                  tokenizer: PreTrainedTokenizer,
#                  prompt_key='prompt',
#                  max_prompt_length=1024,
#                  filter_prompts=True,
#                  cache_dir='~/.cache/verl/rlhf',
#                  chat_template_func=None,
#                  return_raw_chat=False,
#                  truncation='error',
#                  language_column='language',  # 新增：语言字段的名称
#                  content_column='content',  # 新增：原始内容字段
#                  content_en_column='content_en',  # 新增：英文内容字段
#                  content_source_column='content_source',  # 新增：高资源语言内容字段
#                  ):
#         if not isinstance(parquet_files, (List, ListConfig)):
#             parquet_files = [parquet_files]

#         self.parquet_files = parquet_files
#         self.cache_dir = os.path.expanduser(cache_dir)
#         self.tokenizer = tokenizer

#         self.prompt_key = prompt_key
#         self.max_prompt_length = max_prompt_length
#         self.filter_prompts = filter_prompts

#         self.return_raw_chat = return_raw_chat
#         self.chat_template_func = chat_template_func
#         self.truncation = truncation

#         # 新增：语言字段配置
#         self.language_column = language_column
#         self.content_column = content_column
#         self.content_en_column = content_en_column
#         self.content_source_column = content_source_column

#         self._download()
#         self._read_files_and_tokenize()

#     def _download(self):
#         from verl.utils.fs import copy_local_path_from_hdfs
#         for i, parquet_file in enumerate(self.parquet_files):
#             self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

#     def _read_files_and_tokenize(self):
#         dataframes = []
#         for parquet_file in self.parquet_files:
#             # read parquet files and cache
#             dataframe = pd.read_parquet(parquet_file)
#             dataframes.append(dataframe)
#         self.dataframe = pd.concat(dataframes)

#         print(f'original dataset len: {len(self.dataframe)}')

#         # filter out too long prompts
#         tokenizer = self.tokenizer
#         prompt_key = self.prompt_key

#         # nvm if prompt is too long
#         # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
#         #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
#         #                                                      axis=1)]

#         print(f'filter dataset len: {len(self.dataframe)}')

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, item):
#         """
#         Note that we also return the raw_input_ids so that it can be combined with other chat template
#         """
#         row_dict = self.dataframe.iloc[item].to_dict()

#         # 获取样本对应的语言
#         lang = row_dict.get(self.language_column, 'en')  # 默认英文

#         # 选择 content 字段
#         content = row_dict.get(self.content_column, '')
#         content_en = row_dict.get(self.content_en_column, '')
#         content_source = row_dict.get(self.content_source_column, '')

#         # 根据轮次选择合适的 content 字段
#         # 假设这个 `index` 是可以用来决定哪个字段的
#         # 如果你有 `counter` 记录轮次，可以根据轮次选择：
#         counter = row_dict.get("index", 0)

#         # 基于 counter 选择内容：轮次为 0 使用原始内容，轮次为 1 使用英文内容，轮次为 2 使用高资源语言内容
#         if counter == 0:
#             prompt = content
#         elif counter == 1:
#             prompt = content_en
#         elif counter == 2:
#             prompt = content_source
#         else:
#             prompt = content  # 默认返回原内容

#         if self.tokenizer.chat_template:
#             prompt_with_chat_template = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
#         else:
#             prompt_with_chat_template = prompt[0]['content']

#         input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
#                                                                          tokenizer=self.tokenizer,
#                                                                          max_length=self.max_prompt_length,
#                                                                          pad_token_id=self.tokenizer.pad_token_id,
#                                                                          left_pad=True,
#                                                                          truncation=self.truncation)

#         position_ids = compute_position_id_with_mask(attention_mask)

#         row_dict['input_ids'] = input_ids[0]
#         row_dict['attention_mask'] = attention_mask[0]
#         row_dict['position_ids'] = position_ids[0]

#         # encode prompts without chat template
#         if self.return_raw_chat:
#             row_dict['raw_prompt'] = prompt.tolist()

#         # add index for each prompt
#         index = row_dict.get("extra_info", {}).get("index", 0)
#         row_dict["index"] = index

#         return row_dict
