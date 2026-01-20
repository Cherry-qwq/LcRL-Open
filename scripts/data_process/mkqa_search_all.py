import pandas as pd
import os
from collections import OrderedDict

def _ensure_exists(path, lang, kind):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Missing {kind} parquet for language '{lang}': {path}")

def _require_min_len(df, min_len, lang, split):
    n = len(df)
    if n < min_len:
        raise ValueError(
            f"[ERROR] Language '{lang}' {split} set has only {n} rows, "
            f"but requires at least {min_len}."
        )

def sample_and_merge_parquet(
    local_dir,
    languages,
    other_languages=None,
    train_out='train_all.parquet',
    test_out='test_all.parquet',
    test_others_out='test_others.parquet',
    train_n=1000,
    test_n=250,
    seed=42
):
    """
    从各语言 Parquet 文件中采样并合并：
      - 对 `languages`：
        * train_{lang}.parquet: 每种语言采样 train_n 条（不足报错）→ 合并后再打乱
        * test_{lang}.parquet : 每种语言取前 test_n 条（不足报错）→ 顺序合并
      - 对 `other_languages`（仅测试集）：
        * test_{lang}.parquet : 每种语言取前 test_n 条（不足报错）→ 顺序合并并保存为 test_others_out
    """
    # ---------- 语言列表清洗（按原顺序去重） ----------
    def dedupe_keep_order(seq):
        seen = OrderedDict()
        for x in seq:
            if x not in seen:
                seen[x] = True
        return list(seen.keys())

    languages = dedupe_keep_order(languages)
    if other_languages is None:
        other_languages = []
    else:
        other_languages = dedupe_keep_order(other_languages)

    print(f"Main languages: {languages}")
    print(f"Other languages (test only): {other_languages}")

    train_dfs = []
    test_dfs = []
    other_test_dfs = []

    # ---------- 主语言处理：训练与测试 ----------
    for lang in languages:
        train_path = os.path.join(local_dir, f"train_{lang}.parquet")
        test_path  = os.path.join(local_dir, f"test_{lang}.parquet")

        print(f"\nProcessing (main) {lang} ...")
        _ensure_exists(train_path, lang, "train")
        _ensure_exists(test_path,  lang, "test")

        train_df = pd.read_parquet(train_path)
        test_df  = pd.read_parquet(test_path)

        # 要求长度足够
        _require_min_len(train_df, train_n, lang, "train")
        _require_min_len(test_df,  test_n,  lang, "test")

        # 训练采样 + 合并后再打乱
        train_sample = train_df.sample(n=train_n, random_state=seed)
        test_sample  = test_df.head(test_n)  # 不打乱


        # train_sample = train_df
        # # 测试集使用全部样本，保持原顺序
        # test_sample  = test_df

        train_dfs.append(train_sample)
        test_dfs.append(test_sample)

    # 合并训练并二次打乱
    train_all = pd.concat(train_dfs, ignore_index=True)
    train_all = train_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 合并测试（不打乱）
    test_all = pd.concat(test_dfs, ignore_index=True)

    # ---------- 其他语言处理：仅测试 ----------
    if other_languages:
        for lang in other_languages:
            test_path = os.path.join(local_dir, f"test_{lang}.parquet")
            print(f"\nProcessing (others) {lang} ...")
            _ensure_exists(test_path, lang, "test")
            test_df = pd.read_parquet(test_path)
            _require_min_len(test_df, test_n, lang, "test")
            other_test_dfs.append(test_df.head(test_n))
            # other_test_dfs.append(test_df)

        test_others = pd.concat(other_test_dfs, ignore_index=True)
    else:
        test_others = pd.DataFrame()

    # ---------- 输出 ----------
    train_all_path = os.path.join(local_dir, train_out)
    test_all_path  = os.path.join(local_dir, test_out)
    train_all.to_parquet(train_all_path)
    test_all.to_parquet(test_all_path)

    print(f"\n✓ Saved {len(train_all)} training samples to: {train_all_path}")
    print(f"✓ Saved {len(test_all)} test samples to: {test_all_path}")

    if other_languages:
        test_others_path = os.path.join(local_dir, test_others_out)
        test_others.to_parquet(test_others_path)
        print(f"✓ Saved {len(test_others)} other-language test samples to: {test_others_path}")

    print("Done!")

if __name__ == "__main__":
    # 原脚本里的主语言
    languages = ["en", "zh_cn", "ja", "ar", "fi", "ru", "fr", "it"]

    # 你给的 other_languages 列表缺一个逗号，并有重复 "es"
    # 这里做了修正并保留原顺序去重：["es", "pt", "ko", "th", "es"] -> ["es", "pt", "ko", "th"]
    other_languages = ["es", "pt", "ko", "th", "de"]

    local_dir = "./data/mkqa/parallel_3"
    print(f"Script started!")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Target directory: {os.path.abspath(local_dir)}")
    print(f"Directory exists: {os.path.exists(local_dir)}")
    if os.path.exists(local_dir):
        print(f"Files in directory: {os.listdir(local_dir)}")
    print("-" * 50)

    sample_and_merge_parquet(
        local_dir=local_dir,
        languages=languages,
        other_languages=other_languages,
        train_out='train_all.parquet',
        test_out='test_all.parquet',
        test_others_out='test_others.parquet',
        train_n=1000,
        test_n=250,
        seed=42
    )
