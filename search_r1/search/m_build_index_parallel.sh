#!/bin/bash

languages=("de" "es" "fi" "it" "ko" "pt" "ts")


retriever_name=me5
retriever_model=intfloat/multilingual-e5-base
export HF_ENDPOINT=https://hf-mirror.com

for lang in "${languages[@]}"
do
    echo "========================================="
    echo "Processing language: $lang"
    echo "========================================="
    
    corpus_file=data/wikipedia/wiki_chunk100_${lang}.jsonl
    save_dir=data/wikipedia/${lang}
    
    CUDA_VISIBLE_DEVICES=6,7 python search_r1/search/index_builder.py \
        --retrieval_method $retriever_name \
        --model_path $retriever_model \
        --corpus_path $corpus_file \
        --save_dir $save_dir \
        --use_fp16 \
        --max_length 256 \
        --batch_size 512 \
        --pooling_method mean \
        --faiss_type Flat \
        --save_embedding
    
    # 检查上一个命令是否成功
    if [ $? -eq 0 ]; then
        echo "Successfully processed $lang"
    else
        echo "Error processing $lang"
        exit 1
    fi
done

echo "All languages processed successfully!"