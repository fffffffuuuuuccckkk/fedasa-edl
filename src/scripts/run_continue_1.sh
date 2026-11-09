#!/bin/bash

# ==========================================
# 1. 定义指令列表 (数组)
#在这里按顺序放入你需要运行的所有命令
# ==========================================
declare -a COMMAND_LIST=(
    "python train_classifier_ssfl.py --data_name CIFAR100 --model_name wresnet28x2 --control_name 2500_fix@0.95-divcon_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --data_name CIFAR100 --model_name wresnet28x2 --control_name 1250_fix@0.95-divcon_100_0.1_iid_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --data_name CIFAR100 --model_name wresnet28x2 --control_name 2500_fix@0.95-iomatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --data_name CIFAR100 --model_name wresnet28x2 --control_name 1250_fix@0.95-iomatch_100_0.1_iid_5-5_0.5_1_1"
)

# ==========================================
# 2. 开始按顺序处理指令
# ==========================================

# ${#COMMAND_LIST[@]} 获取数组长度
TOTAL_JOBS=${#COMMAND_LIST[@]}
CURRENT_JOB_INDEX=5

for COMMAND in "${COMMAND_LIST[@]}"
do
    ((CURRENT_JOB_INDEX++))

    # 为当前指令生成独立的日志文件
    # 使用时间戳确保文件名不冲突
    LOG_FILE="results/job_${CURRENT_JOB_INDEX}_of_${TOTAL_JOBS}_$(date +%Y%m%d_%H%M%S).log"

    echo "========================================================"
    echo "Starting Job ${CURRENT_JOB_INDEX}/${TOTAL_JOBS}"
    echo "Command: ${COMMAND}"
    echo "Log File: ${LOG_FILE}"
    echo "========================================================"

    # ==========================================
    # 3. 内层循环：失败重试逻辑
    # ==========================================
    while true
    do
        echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Starting/Restarting process..." | tee -a "${LOG_FILE}"

        # 运行当前指令
        # 2>&1 将错误输出合并到标准输出
        # | tee 同时输出到屏幕和文件
        ${COMMAND} 2>&1 | tee -a "${LOG_FILE}"

        # 获取管道中第一个命令(Python)的退出码
        # 注意：必须使用 bash 运行脚本，sh 不支持数组 PIPESTATUS
        EXIT_CODE=${PIPESTATUS[0]}

        echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Process exited with status code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"

        # 判断退出码
        if [ "${EXIT_CODE}" -eq 0 ]; then
            echo "✅ Job ${CURRENT_JOB_INDEX} finished successfully." | tee -a "${LOG_FILE}"
            echo ""
            # 成功则跳出 'while' 循环，继续外层的 'for' 循环执行下一条指令
            break
        else
            RESTART_DELAY=10
            echo "❌ Job ${CURRENT_JOB_INDEX} crashed. Retrying in ${RESTART_DELAY}s..." | tee -a "${LOG_FILE}"
            sleep ${RESTART_DELAY}
            # 这里不 break，继续 while 循环重试同一条指令
        fi
    done
done

echo "All ${TOTAL_JOBS} jobs completed successfully!"