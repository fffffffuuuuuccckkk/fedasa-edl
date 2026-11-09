#!/bin/bash

# ==========================================
# 0. 配置参数
# ==========================================
mkdir -p results
MAX_PARALLEL=4          # 最大同时运行的任务数（根据你的显存/CPU调整）
RESTART_DELAY=15        # 崩溃后的重启延迟（秒）

# ==========================================
# 1. 定义指令列表 (在此处直接指定 --device)
# ==========================================
declare -a COMMAND_LIST=(
    "python train_classifier_ssfl.py --device cuda:0 --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --device cuda:1 --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-mix_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --device cuda:2 --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-openmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --device cuda:3 --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-openmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --device cuda:0 --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-iomatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --device cuda:1 --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-iomatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --device cuda:2 --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-bdmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --device cuda:3 --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-bdmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
)

# 日志名称列表
declare -a LOG_NAMES=(
    new_ours_50_noniid new_ours_25_noniid
    new_openmatch_50_noniid new_openmatch_25_noniid
    new_iomatch_50_noniid new_iomatch_25_noniid
    new_bdmatch_50_noniid new_bdmatch_25_noniid
)

# ==========================================
# 2. 核心任务执行逻辑
# ==========================================
run_with_retry() {
    local cmd=$1
    local name=$2
    local log="results/${name}.log"

    while true; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> Starting: $name" | tee -a "$log"
        echo "Command: $cmd" >> "$log"

        # 直接执行指令，输出全部重定向到日志
        $cmd >> "$log" 2>&1

        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $name finished successfully." | tee -a "$log"
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $name crashed (Code: $EXIT_CODE). Retrying in ${RESTART_DELAY}s..." | tee -a "$log"
            sleep $RESTART_DELAY
        fi
    done
}

# ==========================================
# 3. 进程池管理
# ==========================================
TOTAL_JOBS=${#COMMAND_LIST[@]}
job_idx=0

echo "Parallel Manager started. Total jobs: $TOTAL_JOBS, Max parallel: $MAX_PARALLEL"

while [ $job_idx -lt $TOTAL_JOBS ]; do
    # 统计后台正在运行的任务数量
    running_count=$(jobs -r | wc -l)

    if [ "$running_count" -lt "$MAX_PARALLEL" ]; then
        current_cmd="${COMMAND_LIST[$job_idx]}"
        current_name="${LOG_NAMES[$job_idx]}"

        # 后台异步执行
        run_with_retry "$current_cmd" "$current_name" &

        echo "[Launch] Job $((job_idx+1))/$TOTAL_JOBS: $current_name"
        ((job_idx++))
    fi

    # 稍微休息，避免 CPU 占用过高
    sleep 2
done

# 等待最后一批任务全部完成
wait
echo "All jobs are done!"