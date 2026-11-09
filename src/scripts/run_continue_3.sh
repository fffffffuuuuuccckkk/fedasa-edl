#!/bin/bash

# ==========================================
# 0. 准备工作
# ==========================================
mkdir -p results # 确保结果目录存在，防止报错

# ==========================================
# 1. 定义指令列表 (数组)
# ==========================================
declare -a COMMAND_LIST=(
    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --weight_mode exp --fisher_c 0.5 --seen_class_num 6 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --weight_mode simple --fisher_c 0.5 --seen_class_num 6 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --use_3sigma '' --model_name wresnet28x2 --seen_class_num 6 --fisher_c 0.5 --control_name 300_fix@0.95-mix-edl_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"

    "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --fisher_c 0.5 --control_name 300_fix@0.95-mix-edl_100_0.1_non-iid-d-0.3_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name TinyImageNet --task OSR --model_name wresnet28x2 --seen_class_num 150 --control_name 3750_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --use_3sigma '' --model_name wresnet28x2 --seen_class_num 6 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --kl_w 0.5 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --kl_w 0.1 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --kl_w 0.5 --control_name 300_fix@0.95-mix-edl-randomselectclient_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --kl_w 1 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --fisher_c 1 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --nedl_server 0 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --nedl_client 0 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --pedl_client 0 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --control_name 300_fix@0.95-mix1-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --fisher_c 0 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --nouse_fisher True --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --use_kl 1 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --use_kl 2 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --nouse_kl True --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --adj_fish True --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --control_name 300_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR100 --task OSR --model_name wresnet28x2 --seen_class_num 80 --control_name 4000_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name TinyImageNet --task OSR --model_name wresnet28x2 --seen_class_num 150 --control_name 7500_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR10 --task OSR --model_name wresnet28x2 --seen_class_num 6 --control_name 150_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name CIFAR100 --task OSR --model_name wresnet28x2 --seen_class_num 80 --control_name 2000_fix@0.95-mix-edl_100_0.1_iid_5-5_0.5_1_1"



    # "python train_classifier_ssfl.py --data_name MIOTCD --task None --model_name wresnet28x2 --seen_class_num 6 --control_name 300_fix@0.95-mix_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name MIOTCD --model_name wresnet28x2 --seen_class_num 6 --control_name 300_fix@0.95-iomatch_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name TinyImageNet --task None --model_name wresnet28x2 --seen_class_num 150 --control_name 7500_fix@0.95-mix_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name TinyImageNet --model_name wresnet28x2 --seen_class_num 150 --control_name 7500_fix@0.95-mix_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name TinyImageNet --model_name wresnet28x2 --seen_class_num 150 --control_name 7500_fix@0.95-divcon_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name TinyImageNet --model_name wresnet28x2 --seen_class_num 150 --control_name 7500_fix@0.95-openmatch_100_0.1_iid_5-5_0.5_1_1"

    # "python train_classifier_ssfl.py --data_name TinyImageNet --model_name wresnet28x2 --seen_class_num 150 --control_name 7500_fix@0.95-iomatch_100_0.1_iid_5-5_0.5_1_1"
)

# ==========================================
# 2. 定义对应的日志文件名 (数组)
# 注意：这里的顺序必须和上面的 COMMAND_LIST 一一对应
# ==========================================
declare -a LOG_NAMES=(
    # new_edl_换成exp_fc0.5
    # new_edl_换成倒数_fc0.5
    new_edl_删除自适应_noniid_fc0.5
    new_edl_noniid_fc0.5
    # new_edl_tinyimagenet_25
    # new_edl_删除自适应
    # new_edl_klw_0.5
    # new_edl_klw_0.1
    # new_edl_randomclient
    # new_edl_klw_1
    # new_edl_fc_1

    # new_edl_消融服务器nedl
    # new_edl_消融客户端nedl
    # new_edl_消融客户端pedl
    # new_edl_消融mix
    # new_edl_nofc
    # new_edl_pedl调整fisher
    # new_edl_nedl对齐kl1
    # new_edl_nedl对齐kl2
    # new_edl_nedl删除kl
    # new_edl_删除负向fish_全体动态调整
    # new_edl
    # new_eld_8020
    # new_edl_tinyimagenet

    # new_edl_25
    # new_eld_8020_25

    # new_semifl_MIOTCD
    # new_iomatch_MIOTCD

    # new_semifl_50_TinyImageNet
    # new_ours_50_TinyImageNet
    # new_divcon_50_TinyImageNet
    # new_openmatch_50_TinyImageNet
    # new_iomatch_50_TinyImageNet
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


# 设定参数
RETRY_DELAY_BASE=300  # 失败后至少休息 300 秒 (5分钟)
SUCCESS_REST=0       # 任务成功后休息 60 秒再进行下一个
MAX_RETRIES=999         # 单个任务最大重试次数 (可选)

for (( i=0; i<TOTAL_JOBS; i++ ))
do
    COMMAND="${COMMAND_LIST[$i]}"
    NAME="${LOG_NAMES[$i]}"
    LOG_FILE="results/${NAME}.log"
    JOB_ID=$((i+1))

    RETRY_COUNT=0

    echo "========================================================"
    echo "Starting Job ${JOB_ID}/${TOTAL_JOBS}: ${NAME}"
    echo "========================================================"

    while true
    do
        echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Executing..." | tee -a "${LOG_FILE}"

        # ${COMMAND} 2>&1 | tee -a "${LOG_FILE}"
        eval "${COMMAND}" 2>&1 | tee -a "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ "${EXIT_CODE}" -eq 0 ]; then
            echo "✅ Job ${JOB_ID} finished successfully." | tee -a "${LOG_FILE}"
            echo "--- Cooling down for ${SUCCESS_REST}s before next job ---"
            sleep ${SUCCESS_REST}
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))

            # 如果重试次数过多，建议手动检查，防止服务器爆炸
            if [ ${RETRY_COUNT} -gt ${MAX_RETRIES} ]; then
                echo "❌ Max retries reached for Job ${JOB_ID}. Skipping..." | tee -a "${LOG_FILE}"
                break
            fi

            # --- 新增：非 root 内存碎片清理指令 ---
            echo "🧹 Job crashed. Attempting to clear memory fragments before retry..." | tee -a "${LOG_FILE}"

            # 1. 同步磁盘缓存
            sync

            # 2. 强制 glibc 归还内存给操作系统 (通过 Python 调用 malloc_trim)
            # 这一步不需要 root，能有效处理 Python 程序留下的内存空洞
            python3 -c "import ctypes; ctypes.CDLL('libc.so.6').malloc_trim(0)" 2>/dev/null

            # 3. 清理当前用户下可能残留的僵尸进程
            ps -ef | grep $(whoami) | grep defunct | awk '{print $2}' | xargs -r kill -9 2>/dev/null
            # ---------------------------------------

            # 这里的随机数可以防止多个进程在同一秒重启
            # RANDOM_OFFSET=$(( RANDOM % 60 ))
            RANDOM_OFFSET=0
            TOTAL_WAIT=$(( RETRY_DELAY_BASE + RANDOM_OFFSET ))

            echo "❌ Job crashed (Code: ${EXIT_CODE}). Retry #${RETRY_COUNT} in ${TOTAL_WAIT}s..." | tee -a "${LOG_FILE}"
            sleep ${TOTAL_WAIT}
        fi
    done
done

echo "All jobs processed!"