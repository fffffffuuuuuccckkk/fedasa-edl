#!/bin/bash

# ==========================================
# 0. 准备工作
# ==========================================
mkdir -p results # 确保结果目录存在，防止报错

# ==========================================
# 1. 定义指令列表 (数组)
# ==========================================
declare -a COMMAND_LIST=(
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-bdmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-bdmatch_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-bdmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name SVHN --model_name wresnet28x2 --control_name 300_fix@0.95-bdmatch_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name SVHN --model_name wresnet28x2 --control_name 300_fix@0.95-bdmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name SVHN --model_name wresnet28x2 --control_name 150_fix@0.95-bdmatch_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name SVHN --model_name wresnet28x2 --control_name 150_fix@0.95-bdmatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix1-randomselectclient_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix1-randommixserver_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix-randomselectclient-randommixserver_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix1-randomselectclient-randommixserver_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-bdmatch_100_0.1_non-iid-l-2_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR100 --model_name wresnet28x2 --control_name 4000_fix@0.95-iomatch_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-openmatch_100_0.1_non-iid-l-2_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-iomatch_100_0.1_non-iid-l-2_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-bdmatch_100_0.1_non-iid-l-2_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-mix_100_0.1_non-iid-l-2_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-divcon_100_0.1_non-iid-l-2_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-fedopenmatch_100_0.1_non-iid-l-2_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 150_fix@0.95-mix_100_0.1_non-iid-l-2_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix2_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix1-mix2_100_0.1_iid_5-5_0.5_1_1"
    "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix1-mix2-randomselectclient_100_0.1_iid_5-5_0.5_1_1"
    # "python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix2-randomselectclient_100_0.1_iid_5-5_0.5_1_1"



)

# ==========================================
# 2. 定义对应的日志文件名 (数组)
# 注意：这里的顺序必须和上面的 COMMAND_LIST 一一对应
# ==========================================
declare -a LOG_NAMES=(
    # "bdmatch_noniid"
    # "bdmatch_25"
    # "bdmatch_noniid_25"

    # "bdmatch_svhn"
    # "bdmatch_svhn_noniid"
    # "bdmatch_svhn_25"
    # "bdmatch_svhn_noniid_25"

    # "mixup_randomselect"
    # "mixup_randomserver"
    # "randomselect_randomserver"
    # "ablation_all"
    # "bdmatch_noniid_l_2"
    # "iomatch_8020_noniid"
    # "openmatch_noniid_l_2"
    # "iomatch_noniid_l_2"
    # "bdmatch_noniid_l_2"

    # "semifl_25_l_2"
    # "divcon_25_l_2"
    # "fedopenmatch_25_l_2"
    # "affinity_25_l_2"

    # "mix2"
    # "nomixuploss_mix2"
    "mix2_nomixuploss_randclient"
    # "mix2_randclient"

)

# ==========================================
# 3. 检查配置并开始按顺序处理
# ==========================================

# 获取数组长度
TOTAL_JOBS=${#COMMAND_LIST[@]}
TOTAL_NAMES=${#LOG_NAMES[@]}

# 简单检查两个数组长度是否一致
if [ "$TOTAL_JOBS" -ne "$TOTAL_NAMES" ]; then
    echo "Error: 指令数量 ($TOTAL_JOBS) 与 日志名数量 ($TOTAL_NAMES) 不匹配！"
    exit 1
fi

# 使用 C 语言风格的 for 循环，通过索引 i 同时获取命令和日志名
for (( i=0; i<TOTAL_JOBS; i++ ))
do
    # 获取当前索引的内容
    COMMAND="${COMMAND_LIST[$i]}"
    NAME="${LOG_NAMES[$i]}"

    # 构造日志文件路径 (这里手动指定了文件名)
    LOG_FILE="results/${NAME}.log"

    JOB_ID=$((i+1))

    echo "========================================================"
    echo "Starting Job ${JOB_ID}/${TOTAL_JOBS}"
    echo "Name:    ${NAME}"
    echo "Command: ${COMMAND}"
    echo "Log File: ${LOG_FILE}"
    echo "========================================================"

    # ==========================================
    # 4. 内层循环：失败重试逻辑
    # ==========================================
    while true
    do
        echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Starting/Restarting process..." | tee -a "${LOG_FILE}"

        # 运行当前指令
        # 2>&1 将错误输出合并到标准输出
        # | tee -a 同时输出到屏幕和文件 (追加模式)
        ${COMMAND} 2>&1 | tee -a "${LOG_FILE}"

        # 获取管道中第一个命令(Python)的退出码
        EXIT_CODE=${PIPESTATUS[0]}

        echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Process exited with status code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"

        # 判断退出码
        if [ "${EXIT_CODE}" -eq 0 ]; then
            echo "✅ Job ${JOB_ID} (${NAME}) finished successfully." | tee -a "${LOG_FILE}"
            echo ""
            # 成功则跳出 'while' 循环，进入下一轮 'for' 循环
            break
        else
            RESTART_DELAY=10
            echo "❌ Job ${JOB_ID} (${NAME}) crashed. Retrying in ${RESTART_DELAY}s..." | tee -a "${LOG_FILE}"
            sleep ${RESTART_DELAY}
            # 这里不 break，继续 while 循环重试同一条指令
        fi
    done
done

echo "All ${TOTAL_JOBS} jobs completed successfully!"