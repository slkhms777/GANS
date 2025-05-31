#!/bin/bash

# 设置脚本在遇到错误时退出
set -e

# 创建日志目录和时间戳
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="./logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 设置Python路径
export PYTHONPATH="$PWD:$PYTHONPATH"

# 定义日志文件路径
TRAINING_LOG="${LOG_DIR}/training.log"
ERROR_LOG="${LOG_DIR}/error.log"
SYSTEM_LOG="${LOG_DIR}/system.log"

# 打印信息到控制台和日志
echo "=== MiniBigGAN 训练开始 ===" | tee "$SYSTEM_LOG"
echo "开始时间: $(date)" | tee -a "$SYSTEM_LOG"
echo "日志目录: $LOG_DIR" | tee -a "$SYSTEM_LOG"

# 记录系统信息
echo "=== 系统信息 ===" >> "$SYSTEM_LOG"
echo "Python版本: $(python --version)" >> "$SYSTEM_LOG"
echo "工作目录: $(pwd)" >> "$SYSTEM_LOG"
echo "PYTHONPATH: $PYTHONPATH" >> "$SYSTEM_LOG"

# 检查CUDA和GPU信息
echo "=== GPU信息 ===" >> "$SYSTEM_LOG"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" >> "$SYSTEM_LOG" 2>&1

# nvidia-smi信息（如果可用）
if command -v nvidia-smi &> /dev/null; then
    echo "=== nvidia-smi 信息 ===" >> "$SYSTEM_LOG"
    nvidia-smi >> "$SYSTEM_LOG" 2>&1
fi

echo "开始训练..." | tee -a "$SYSTEM_LOG"

# 运行训练，同时输出到控制台和保存到日志文件
# 使用tee同时输出到控制台和日志文件
# 使用exec重定向stderr到error.log
exec 2> >(tee -a "$ERROR_LOG" >&2)

python main.py 2>&1 | tee "$TRAINING_LOG"

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "训练成功完成!" | tee -a "$SYSTEM_LOG"
    echo "结束时间: $(date)" | tee -a "$SYSTEM_LOG"
    
    # 创建训练总结
    SUMMARY_FILE="${LOG_DIR}/training_summary.txt"
    echo "=== 训练总结 ===" > "$SUMMARY_FILE"
    echo "开始时间: $(head -2 "$SYSTEM_LOG" | tail -1)" >> "$SUMMARY_FILE"
    echo "结束时间: $(date)" >> "$SUMMARY_FILE"
    echo "日志目录: $LOG_DIR" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "生成的文件:" >> "$SUMMARY_FILE"
    ls -la checkpoints/ >> "$SUMMARY_FILE" 2>/dev/null || echo "无检查点文件" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    ls -la visual/ >> "$SUMMARY_FILE" 2>/dev/null || echo "无可视化文件" >> "$SUMMARY_FILE"
    
    echo "训练总结已保存到: $SUMMARY_FILE"
else
    echo "训练过程中发生错误!" | tee -a "$SYSTEM_LOG"
    echo "错误时间: $(date)" | tee -a "$SYSTEM_LOG"
    echo "请检查错误日志: $ERROR_LOG"
    echo "请检查训练日志: $TRAINING_LOG"
    exit 1
fi

echo "所有日志已保存到: $LOG_DIR"
echo "- 训练日志: $TRAINING_LOG"
echo "- 错误日志: $ERROR_LOG"  
echo "- 系统日志: $SYSTEM_LOG"
echo "训练完成!"