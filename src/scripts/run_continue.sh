#!/bin/bash

# 定义要运行的 Python 命令
# 请将 'python' 替换为你的 Python 解释器路径 (如 /path/to/venv/bin/python)
# 并确保命令参数正确。
COMMAND="python train_classifier_ssfl.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 300_fix@0.95-mix_100_0.1_iid_5-5_0.5_1_1"

# 定义日志文件，用于记录程序输出
LOG_FILE="results/training_output_$(date +%Y%m%d_%H%M%S).log"

# 无限循环，用于持续运行和重启
echo "Starting continuous run. Output is being redirected to ${LOG_FILE}"

# 循环直到用户手动停止脚本
while true
do
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Starting/Restarting training process..." | tee -a "${LOG_FILE}"

    # 运行命令，并将标准输出和标准错误追加到日志文件
    # 2>&1 表示将标准错误(2)重定向到标准输出(1)
    # tee -a 允许同时在终端打印输出，并将输出追加到文件
    # 注意：为了让tee也输出到日志文件，需要用( )子shell来运行COMMAND
    ( ${COMMAND} ) 2>&1 | tee -a "${LOG_FILE}"

    # $? 捕获上一个命令的退出状态。0 表示成功退出，非 0 通常表示错误或中断。
    EXIT_CODE=$?

    echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Training process exited with status code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"

    # 如果需要判断特定的退出码才重启，可以在这里修改
    # if [ ${EXIT_CODE} -ne 0 ]; then
    #   ...
    # fi

    # 每次重启前等待一段时间，避免因为瞬时错误导致快速循环重启
    RESTART_DELAY=10
    echo "Waiting for ${RESTART_DELAY} seconds before next restart attempt..." | tee -a "${LOG_FILE}"
    sleep ${RESTART_DELAY}
done