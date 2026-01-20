#!/bin/bash

file_path=data/wikipedia
retriever_name=me5
retriever_path=intfloat/multilingual-e5-base

# 定义语言列表
# languages=("de" "es" "fi" "it" "ko" "pt" "th" "en" "fr" "zh" "ja" "ar" "ru")

# 起始端口
start_port=8003

# 日志目录
log_dir="logs/retrieval_servers"
mkdir -p $log_dir

# 循环启动每个语言的服务器
for i in "${!languages[@]}"; do
    lang="${languages[$i]}"
    port=$((start_port + i))
    
    # 设置文件路径
    index_file=$file_path/$lang/me5_Flat.index
    corpus_file=$file_path/wiki_chunk100_${lang}.jsonl
    
    # 日志文件
    log_file=$log_dir/retriever_${lang}_${port}.log
    
    echo "Starting retriever for language: $lang on port: $port"
    
    # 后台启动服务
    CUDA_VISIBLE_DEVICES=6,7 nohup python search_r1/search/retrieval_server_parallel.py \
        --index_path $index_file \
        --corpus_path $corpus_file \
        --topk 3 \
        --retriever_name $retriever_name \
        --retriever_model $retriever_path \
        --faiss_gpu \
        --port $port \
        --language $lang \
        > $log_file 2>&1 &
    
    # 保存进程ID
    echo $! > $log_dir/retriever_${lang}_${port}.pid
    
    echo "  - PID: $!"
    echo "  - Log: $log_file"
    echo ""
    
    # 稍微延迟，避免同时启动造成资源竞争
    sleep 2
done

echo "All retrieval servers started!"
echo ""
echo "Port mapping:"
for i in "${!languages[@]}"; do
    lang="${languages[$i]}"
    port=$((start_port + i))
    echo "  $lang: http://0.0.0.0:$port"
done

echo ""
echo "To check status: ps aux | grep retrieval_server_parallel.py"
echo "To stop all servers:pkill -f "retrieval_server_parallel.py""