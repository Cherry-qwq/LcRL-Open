import pandas as pd
import requests
import json
import argparse
import sys
import torch
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import AutoTokenizer

LANG_NAME_MAP = {
    'en': 'English', 'zh_cn': 'Chinese', 'zh': 'Chinese', 'ja': 'Japanese',
    'ar': 'Arabic', 'fi': 'Finnish', 'ru': 'Russian', 'fr': 'French',
    'it': 'Italian', 'pt': 'Portuguese', 'es': 'Spanish', 'ko': 'Korean',
    'de': 'German', 'th': 'Thai'
}

class RAGDebugger:
    def __init__(self, model_path: str, retriever_base_url: str):
        print(f"[*] 正在加载分词器: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.base_url = retriever_base_url.rstrip("/")
        
        self.lang_to_port = {
            'en': 8000, 'zh_cn': 8001, 'zh': 8001, 'ja': 8002,
            'ar': 8003, 'fi': 8004, 'ru': 8005, 'fr': 8006,
            'it': 8007, 'pt': 8008, 'es': 8009, 'ko': 8010,
            'de': 8011, 'th': 8012,
        }
        
        self.high_resource_map = {
            'fr': 'it', 'es': 'fr', 'pt': 'fr', 'it': 'fr', 'de': 'fr',
            'en': 'fr', 'ru': 'fr', 'zh_cn': 'ja', 'zh': 'ja', 'ja': 'zh',
            'ko': 'ja', 'th': 'ja', 'fi': 'ru', 'ar': 'fr',
        }

    def _get_target_langs(self, lang: str) -> List[str]:
        lang = lang.lower().replace('zh-cn', 'zh_cn')
        high_A = self.high_resource_map.get(lang)
        high_B = self.high_resource_map.get('en')
        high_C = self.high_resource_map.get(high_A) if high_A else None

        candidates = [lang, 'en', high_A, high_B, high_C]
        seen = []
        for c in candidates:
            if c and c not in seen: seen.append(c)
        
        fallbacks = ['fr', 'it', 'zh', 'ja', 'ru', 'ar', 'fi']
        for f in fallbacks:
            if len(seen) >= 5: break
            if f not in seen: seen.append(f)
        return seen[:5]

    def _retrieve(self, query: str, lang: str) -> str:
        port = self.lang_to_port.get(lang)
        if not port: return ""
        
        url = f"{self.base_url}:{port}/retrieve"
        payload = {"queries": [query], "topk": 2, "return_scores": True}
        
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            res_json = resp.json()
            results = res_json.get("result", [[]])[0]
            
            if not results: return ""
            
            docs = []
            for item in results:
                content = item.get('document', {}).get('contents', '')
                tokens = self.tokenizer.encode(content, add_special_tokens=False, truncation=True, max_length=200)
                text = self.tokenizer.decode(tokens)
                docs.append(f"[{lang}] {text}")
            return "\n".join(docs)
        except Exception as e:
            # 这里的打印是安全的，不包含反斜杠处理
            print(f"\n[!] 端口 {port} ({lang}) 连接失败或返回异常: {str(e)}")
            return ""

    def clean_output(self, rm: any) -> str:
        """严格提取 ground_truth，确保输出为 "output": "value" 格式"""
        if not isinstance(rm, dict) or "ground_truth" not in rm:
            return ""
        
        gt = rm["ground_truth"].get("target", "")
        
        # 处理多种可能的类型 (list, numpy array, tensor)
        if hasattr(gt, "__iter__") and not isinstance(gt, str):
            gt_list = list(gt)
            gt = gt_list[0] if len(gt_list) > 0 else ""
        
        # 处理字符串形式的 ['ans']
        res = str(gt).strip()
        if res.startswith("['") and res.endswith("']"):
            res = res[2:-2]
        elif res.startswith("[") and res.endswith("]"):
            res = res[1:-1].replace("'", "").replace('"', "")
            
        return res

    def process(self, input_path: str, output_path: str):
        df = pd.read_parquet(input_path)
        final_json = []

        print(f"[*] 数据加载成功，总计: {len(df)} 条")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving"):
            q = row['question']
            lang_code = row['language'].lower().replace('zh-cn', 'zh_cn')
            target_langs = self._get_target_langs(lang_code)
            
            info_list = []
            for l in target_langs:
                doc_text = self._retrieve(q, l)
                if doc_text.strip():
                    info_list.append(doc_text)
            
            combined_info = "\n".join(info_list)
            
            # --- 强制报错逻辑 ---
            if not combined_info.strip():
                print(f"\n[CRITICAL ERROR] 样本 {idx} 无法获取任何检索内容！")
                print(f"Query: {q}")
                print(f"Target Ports: {[self.lang_to_port.get(l) for l in target_langs]}")
                raise RuntimeError("检测到空检索结果。请确认检索服务器已启动且端口映射正确。")

            # --- 修复 SyntaxError: 移出 f-string 处理 ---
            if idx < 3:
                preview_text = combined_info[:100].replace('\n', ' ')
                print(f"\n[DEBUG {idx}] 检索成功预览: {preview_text}...")
            
            lang_name = LANG_NAME_MAP.get(lang_code, "English")
            
            final_json.append({
                "instruction": f"Answer the given question in {lang_name}. You can directly provide the answer without detailed illustrations.",
                "input": f"Question: {q}\nInformation: {combined_info}",
                "output": self.clean_output(row['reward_model'])
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)
        
        print(f"\n[√] 处理完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/slow_share/huggingface/Qwen/Qwen3-8B")
    parser.add_argument("--input", default="data/mkqa/analysis/mkqa_train_1lang_8000.parquet")
    parser.add_argument("--output", default="data/mkqa/sft_mkqa_1lang_8000.json")
    parser.add_argument("--url", default="http://127.0.0.1")
    args = parser.parse_args()

    debugger = RAGDebugger(args.model_path, args.url)
    try:
        debugger.process(args.input, args.output)
    except RuntimeError as e:
        print(f"\n程序已强行终止: {e}")
        sys.exit(1)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_file", default="data/xor/xor_train_all.parquet")
#     parser.add_argument("--output_file", default="data/xor/sft_xor.json")
#     parser.add_argument("--model_path", default="/slow_share/ruiqi/huggingface/models/Qwen/Qwen2.5-3B-Instruct")
#     args = parser.parse_args()

#     gen = DatasetGenerator(args.model_path)
#     gen.run(args.input_file, args.output_file)