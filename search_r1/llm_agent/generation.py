import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import random
@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url_base: str = "http://127.0.0.1"
    topk: int = 3

    language_to_port: Dict[str, int] = None
    language_to_high_resource: Dict[str, str] = None
    def __post_init__(self):
        if self.language_to_port is None:
            self.language_to_port = {
                'en': 8000,
                'zh_cn': 8001,
                'zh': 8001,
                'ja': 8002,
                'ar': 8003,
                'fi': 8004,
                'ru': 8005,
                'fr': 8006,
                'it': 8007,
                'pt': 8008,
                'es': 8009,
                'ko': 8010,
                'de': 8011,
                'th': 8012,
                'bn': 8013,
                'te': 8014,
            }
        
        if self.language_to_high_resource is None:
            self.language_to_high_resource = {
                'fr': 'it',  
                'es': 'fr',   
                'pt': 'fr',  
                'it': 'fr',   
                'de': 'fr',   
                'en': 'fr',   
                'ru': 'fr',  
                'zh_cn': 'ja',
                'zh': 'ja',
                'ja': 'zh',   
                'ko': 'ja',   
                'th': 'ja',   
                'fi': 'ru',  
                'ar': 'fr', 
                'bn': 'fr', 
                'te': 'fr', 
            }


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        self.search_round_tracker = {}  # {sample_idx: current_round}

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('no_think_rl not supported in multi-language search')
            
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding with info mask."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Wrapper for generation that handles multi-GPU padding requirements."""
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, 
                     question_languages: List[str] = None) -> Tuple[Dict, Dict]:
        """
        Run main LLM generation loop with multi-round multi-language search.
        
        Args:
            gen_batch: Generation batch
            initial_input_ids: Initial input IDs
            question_languages: List of language codes for each question in batch
        """
        batch_size = initial_input_ids.shape[0]
        

        if question_languages is None:
            question_languages = ['en'] * batch_size 
            

        self.search_round_tracker = {i: 0 for i in range(batch_size)}
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        round1_search_stats = torch.zeros(batch_size, dtype=torch.int)  
        round2_search_stats = torch.zeros(batch_size, dtype=torch.int)  
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
                
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)


            active_languages = [question_languages[i] for i, m in enumerate(active_mask) if m]
            active_indices = [i for i, m in enumerate(active_mask) if m]
            prev_rounds = {i: self.search_round_tracker.get(i, 0) for i, m in enumerate(active_mask) if m}

            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, 
                self.tokenizer.pad_token, 
                active_mask,
                question_languages=active_languages,
                sample_indices=active_indices
            )
            for idx in active_indices:
                if is_search[idx]:
                    r = prev_rounds.get(idx, 0)
                    if r == 0:
                        round1_search_stats[idx] += 1
                    elif r == 1:
                        round2_search_stats[idx] += 1
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask & curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            


            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            active_languages = [question_languages[i] for i, m in enumerate(active_mask) if m]
            active_indices = [i for i, m in enumerate(active_mask) if m]
            prev_rounds_2 = {i: self.search_round_tracker.get(i, 0) for i, m in enumerate(active_mask) if m}
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, 
                self.tokenizer.pad_token, 
                active_mask, 
                do_search=False,
                question_languages=active_languages,
                sample_indices=active_indices
            )
            for idx in active_indices:
                if is_search[idx]:
                    r = prev_rounds_2.get(idx, 0)
                    if r == 0:
                        round1_search_stats[idx] += 1
                    elif r == 1:
                        round2_search_stats[idx] += 1
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask & curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['round1_search_stats'] = round1_search_stats.tolist() 
        meta_info['round2_search_stats'] = round2_search_stats.tolist()  
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        print(f"Round 1 searches: {round1_search_stats.sum().item()}, Round 2 searches: {round2_search_stats.sum().item()}")
        try:
            if hasattr(gen_batch, "non_tensor_batch") and isinstance(gen_batch.non_tensor_batch, dict):
                langs = gen_batch.non_tensor_batch.get("language", None)
                if langs is not None:
                 
                    meta_info["languages"] = [str(x) for x in langs]
        except Exception as _e:

            pass

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    
    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, 
                            do_search=True, question_languages: List[str] = None,
                            sample_indices: List[int] = None):
        """
        Returns:
            Tuple of (next_obs, dones, valid_action, is_search)
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []

        lang_by_idx = {}
        if question_languages and sample_indices:
            def _norm_lang(s: str) -> str:
                s = (s or 'en').lower().replace('_', '-')
                if s in ('zh-cn', 'zh_cn', 'zh-hans', 'zhs'): return 'zh'
                return s.split('-')[0]
            lang_by_idx = {idx: _norm_lang(lang) for idx, lang in zip(sample_indices, question_languages)}

        search_queries = []
        search_languages = []
        search_sample_indices = []

        for i, (action, content, active) in enumerate(zip(cur_actions, contents, active_mask)):
            if active and action == 'search':
                search_queries.append(content)

                lang = lang_by_idx.get(i, 'en') if lang_by_idx else 'en'
                search_languages.append(lang)
                search_sample_indices.append(i)

        if do_search and len(search_queries) > 0:
            search_results = self.batch_multi_language_search(
                search_queries,
                search_languages,
                search_sample_indices
            )
        else:
            search_results = [''] * len(search_queries)

        search_idx = 0
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results[search_idx].strip()}</information>\n\n')
                    search_idx += 1
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(
                        '\nMy previous action is invalid. If I want to search, I should put the query between '
                        '<search> and </search>. If I want to give the final answer, I should put the answer '
                        'between <answer> and </answer>. Let me try again.\n'
                    )
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert search_idx == len(search_results), f"Search result mismatch: {search_idx} vs {len(search_results)}"
        return next_obs, dones, valid_action, is_search


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """Process predictions into actions and contents."""
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str):
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents
    def _get_high_resource_lang(self, lang_code):
        if not lang_code:
            return None
        lang_code = self._norm_lang(lang_code)
        return self.config.language_to_high_resource.get(lang_code, None)
    
    def batch_multi_language_search(self, queries: List[str], languages: List[str],
                                    sample_indices: List[int]) -> List[str]:
        """
        Perform batch search with language-specific and multi-round logic.
        
        Args:
            queries: Search queries
            languages: Language code for each query
            sample_indices: Original sample indices for tracking search rounds
            
        Returns:
            List of formatted search results
        """
        results_list = []
        
        for query, lang, sample_idx in zip(queries, languages, sample_indices):

            current_round = self.search_round_tracker.get(sample_idx, 0)
            
            if current_round == 0:

                result = self._single_search(query, lang, tk=3)
                results_list.append(result)

                self.search_round_tracker[sample_idx] = 1


            elif current_round == 1:


                orig_lang = self._norm_lang(lang)


                candidate_langs = ['en', 'fr', 'it', 'zh', 'ja', 'ru', 'ar', 'fi']


                search_langs = []
                for l in candidate_langs:
                    if not l:
                        continue
                    norm_l = self._norm_lang(l)
                    if norm_l and norm_l not in search_langs and norm_l != orig_lang:
                        search_langs.append(norm_l)


                multi_results = []
                for l in search_langs:
                    res = self._single_search(query, l)  
                    multi_results.append(f"[lang={l}]\n{res}")

                combined_result = "\n\n".join(multi_results)

                results_list.append(combined_result)
                self.search_round_tracker[sample_idx] = 2


            
                
            else:
            
                result = self._single_search(query, 'en', tk=3)
                results_list.append(result)
        
        return results_list

    def _norm_lang(self, s: str) -> str:
            s = (s or 'en').lower().replace('_','-')
            if s in ('zh-cn','zh_cn','zh-hans','zhs'): return 'zh'
            return s.split('-')[0]
    
    def _single_search(self, query: str, language: str, tk= None ) -> str:
        """
        Perform single search with specific language retriever.
        
        Args:
            query: Search query
            language: Language code
            
        Returns:
            Formatted search result string
        """
        if tk:
            topk = tk
        else:
            topk = self.config.topk
        language = self._norm_lang(language)
        port = self.config.language_to_port.get(language, 8000)
        url = f"{self.config.search_url_base}:{port}/retrieve"
        
        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()['result'][0]
            return self._passages2string(result, language)
        except Exception as e:
            print(f"[ERROR] Search failed for language {language} at {url}: {e}")
            return f"[Search failed for {language}]"

    def _passages2string(self, retrieval_result, language: str = None):
        """Format retrieval results with language annotation."""
        format_reference = ''
        MAX_CHARS_PER_DOC = 200
        
        lang_prefix = f"[{language.upper()}] " if language else ""
        
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            
            if len(content) > MAX_CHARS_PER_DOC:
                content = content[:MAX_CHARS_PER_DOC]
            
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"{lang_prefix}Doc {idx+1}(Title: {title}) {text}\n"
        
        return format_reference