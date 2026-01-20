import os
import sys
import argparse
import glob
import pandas as pd

# ---- 语言名 & 示例答案映射（兼容你的旧脚本；补充了 zh/zh_cn 两个键） ----
LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'zh': 'Chinese',      # 补充
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

EXAMPLE_ANSWERS = {
    'en': 'Beijing',
    'fr': 'l’arctique',
    'es': 'El ártico',
    'de': 'arktis',
    'zh': '北京',         # 补充
    'zh_cn': '北京',
    'ja': '東京',
    'ar': 'القاهرة',
    'ru': 'Москва',
    'pt': 'O pólo norte',
    'it': 'artico',
    'ko': '서울',
    'hi': 'दिल्ली',
    'tr': 'Ankara',
    'fi': 'Helsinki',     # 可按需调整
}

def build_new_prompt(lang_code: str, question: str) -> str:
    """根据新模板生成 prompt 字符串"""
    lang_name = LANGUAGE_NAMES.get(lang_code, 'English')
    example_answer = EXAMPLE_ANSWERS.get(lang_code, 'Answer')
    question = (question or '').strip()

    # 使用你给定的新模板（保持原样，不擅自修正 'your' 拼写）
    prompt = (
        f"Answer the given question. "
        f"You must conduct reasoning inside <think> and </think> first every time you get new information. "
        f"After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
        f"You can search as many times as you want. "
        f"If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
        f"Note: Answers should be in {lang_name}. For example, <answer> {example_answer} </answer>. \n"
        f"Question: {question}\n"
    )
    return prompt

def list_parquet_files(target: str):
    """返回需要处理的 parquet 文件列表（支持文件或目录）"""
    if os.path.isdir(target):
        # 目录：抓取目录下 *.parquet（不递归，如果要递归可用 **/*.parquet, recursive=True）
        return sorted(glob.glob(os.path.join(target, "*.parquet")))
    elif os.path.isfile(target):
        return [target]
    else:
        # 支持简单通配符
        files = sorted(glob.glob(target, recursive=True))
        return [f for f in files if f.endswith(".parquet")]

def process_one_file(in_path: str, out_path: str, inplace: bool = False):
    print(f"[INFO] Loading: {in_path}")
    df = pd.read_parquet(in_path)

    # 基本字段检查
    if "language" not in df.columns:
        raise ValueError(f"{in_path} 缺少必须列 'language'")
    if "question" not in df.columns:
        raise ValueError(f"{in_path} 缺少必须列 'question'")
    if "prompt" not in df.columns:
        raise ValueError(f"{in_path} 缺少必须列 'prompt'（期望为 list[{{role, content}}]）")

    # 构造新 prompt，并回写到 prompt 列（保持结构：list[{"role":"user","content": "..."}]）
    def _rewrite_row(row):
        new_content = build_new_prompt(str(row["language"]), str(row["question"]))
        return [{"role": "user", "content": new_content}]

    df["prompt"] = df.apply(_rewrite_row, axis=1)

    # 保存
    save_path = out_path
    if save_path is None:
        # 自动生成新文件名：xxx.parquet -> xxx.reprompt.parquet
        root, ext = os.path.splitext(in_path)
        save_path = f"{root}.reprompt{ext}"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_parquet(save_path, index=False)
    print(f"[OK] Saved: {save_path}  (rows={len(df)})")

def main():
    ap = argparse.ArgumentParser(description="Rewrite prompt field in parquet files to the new template")
    ap.add_argument("--target", type=str, default="data/xor/*_all.parquet",
                    help="要处理的目标：可以是单个文件、目录，或通配符（如 data/mkqa/parallel/*.parquet）")
    ap.add_argument("--out_dir", type=str, default="data/xor/*_all_naive.parquet",
                    help="输出目录（不指定则在原地生成 *.reprompt.parquet；--inplace 则覆盖原文件）")
    ap.add_argument("--inplace", action="store_true",
                    help="直接覆盖原 parquet 文件（谨慎使用）")
    args = ap.parse_args()

    files = list_parquet_files(args.target)
    if not files:
        print(f"[WARN] 未找到 parquet 文件：{args.target}")
        sys.exit(1)

    if args.inplace and args.out_dir:
        print("[ERROR] --inplace 与 --out_dir 不能同时使用")
        sys.exit(1)

    for fp in files:
        if args.inplace:
            process_one_file(fp, out_path=None, inplace=True)
        else:
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                out_path = os.path.join(args.out_dir, os.path.basename(fp))
            else:
                out_path = None  # 走默认 *.reprompt.parquet
            process_one_file(fp, out_path=out_path, inplace=False)

if __name__ == "__main__":
    main()
