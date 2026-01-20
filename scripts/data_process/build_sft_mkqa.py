# -*- coding: utf-8 -*-
"""
Build SFT training data (sft_mkqa.jsonl) from:
- Retrieval corpus A.jsonl
- MKQA training set B.jsonl

Author: ChatGPT
"""

import json
from typing import List, Dict, Optional

# =========================
# 1. 配置区
# =========================

A_JSONL_PATH = "data/mkqa_all_v12.jsonl"
B_JSONL_PATH = "data/mkqa/parallel_withid/train_all.jsonl"
OUTPUT_PATH = "data/mkqa/sft_mkqa.jsonl"

TOP_K = 2                 # 每种语言取前 top-k 个文档
MAX_TOKENS_PER_DOC = 200  # 每个文档最大 token 数（近似）


# =========================
# 2. 语言映射与选择逻辑
# =========================

language_to_high_resource = {
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
}


def _norm_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = lang.lower()
    if lang in ["zh-cn", "zh_cn", "zh-hans"]:
        return "zh"
    return lang


def _get_high_resource_lang(lang: str) -> Optional[str]:
    lang = _norm_lang(lang)
    return language_to_high_resource.get(lang)


def _get_target_langs(orig_lang: str) -> List[str]:
    """
    根据原始语言，计算出包含原始语言在内的 5 种检索语言列表
    """
    orig_lang = _norm_lang(orig_lang)
    if not orig_lang:
        return ['en']

    high_resource_lang_A = _get_high_resource_lang(orig_lang)
    high_resource_lang_B = _get_high_resource_lang('en')
    high_resource_lang_C = (
        _get_high_resource_lang(high_resource_lang_A)
        if high_resource_lang_A else None
    )

    candidate_langs = ['en', high_resource_lang_A,
                       high_resource_lang_B, high_resource_lang_C]

    search_langs = []
    search_langs.append(orig_lang)

    for l in candidate_langs:
        if not l:
            continue
        l = _norm_lang(l)
        if l and l not in search_langs:
            search_langs.append(l)

    fallback_langs = ['fr', 'it', 'zh', 'ja', 'ru', 'ar', 'fi']
    target_total_langs = 5

    if len(search_langs) < target_total_langs:
        for fb in fallback_langs:
            fb = _norm_lang(fb)
            if fb not in search_langs:
                search_langs.append(fb)
                if len(search_langs) >= target_total_langs:
                    break

    return search_langs[:target_total_langs]


# =========================
# 3. 文本截断工具
# =========================

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """
    简单 token 近似：按空格切分
    """
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


# =========================
# 4. 主流程
# =========================

def main():
    print("Loading A.jsonl ...")
    a_index: Dict[int, dict] = {}

    with open(A_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            a_index[row["example_id"]] = row

    print(f"Loaded {len(a_index)} entries from A.jsonl")

    print("Processing B.jsonl ...")
    sft_samples = []
    missing_cnt = 0

    with open(B_JSONL_PATH, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            b = json.loads(line)
            example_id = b.get("example_id")

            if example_id not in a_index:
                missing_cnt += 1
                continue

            a = a_index[example_id]
            orig_lang = b.get("language", "en")

            target_langs = _get_target_langs(orig_lang)

            docs_blocks = []

            for lang in target_langs:
                key = f"retrieved_docs_{lang}"
                if key not in a:
                    continue

                docs = a[key][:TOP_K]
                if not docs:
                    continue

                docs_blocks.append(f"[{lang}]")
                for i, d in enumerate(docs, 1):
                    content = truncate_by_tokens(
                        d.get("content", ""),
                        MAX_TOKENS_PER_DOC
                    )
                    docs_blocks.append(f"({i}) {content}")

            if not docs_blocks:
                continue

            instruction = (
                f"Answer the given question in {orig_lang.upper()} "
                f"based on the information I provide. "
                f"You can directly provide the answer without detailed illustrations."
            )

            input_text = (
                f"Question:\n{b['question']}\n\n"
                f"Information:\n" + "\n".join(docs_blocks)
            )

            golden_answers = b.get("golden_answers", [])
            if not golden_answers:
                continue

            output_text = golden_answers[0]["text"]

            sft_samples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })

            if line_idx % 1000 == 0:
                print(f"Processed {line_idx} lines...")

    print(f"Missing example_id in A.jsonl: {missing_cnt}")

    print("Writing sft_mkqa.jsonl ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in sft_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Generated {len(sft_samples)} SFT samples.")
    print(f"Saved to: {OUTPUT_PATH}")


# =========================
# 5. 入口
# =========================

if __name__ == "__main__":
    main()
