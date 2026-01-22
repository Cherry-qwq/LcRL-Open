#!/bin/bash

file_path=data/wikipedia
retriever_name=me5
retriever_path=intfloat/multilingual-e5-base


languages=("de" "es" "fi" "it" "ko" "pt" "th" "en" "fr" "zh" "ja" "ar" "ru")


start_port=8003


log_dir="logs/retrieval_servers"
mkdir -p $log_dir


for i in "${!languages[@]}"; do
    lang="${languages[$i]}"
    port=$((start_port + i))
    

    index_file=$file_path/$lang/me5_Flat.index
    corpus_file=$file_path/wiki_chunk100_${lang}.jsonl
    

    log_file=$log_dir/retriever_${lang}_${port}.log
    
    echo "Starting retriever for language: $lang on port: $port"
    

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
    

    echo $! > $log_dir/retriever_${lang}_${port}.pid
    
    echo "  - PID: $!"
    echo "  - Log: $log_file"
    echo ""
    

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