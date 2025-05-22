#!/bin/bash
# filepath: /mnt/16T/gjx/GANS/train_cgan_ddp.sh

# 创建日志目录
LOG_DIR="./Log/terminal"
mkdir -p $LOG_DIR

# 获取当前时间作为日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/cgan_cifar10_ddp_${TIMESTAMP}.log"

# 打印GPU信息到日志
echo "=== GPU信息 ===" | tee -a $LOG_FILE
nvidia-smi | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 设置环境变量（可选）
# 如果需要指定使用哪些GPU，取消下面的注释并设置值
# export CUDA_VISIBLE_DEVICES="0,1"

echo "开始DDP训练，使用2张GPU" | tee -a $LOG_FILE
echo "训练日志将保存到: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 运行训练脚本，并将输出重定向到日志文件
# --nproc_per_node=2 指定使用2个GPU
{
  # 先删除之前可能存在的临时文件
  rm -f /tmp/torch_distributed_*
  
  # 执行torchrun命令
  torchrun --nproc_per_node=2 GAN/CGAN-para.py
} 2>&1 | tee -a $LOG_FILE

# 检查训练是否成功完成
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "训练成功完成！" | tee -a $LOG_FILE
else
  echo "训练过程中出现错误，请检查日志。" | tee -a $LOG_FILE
fi

echo "日志已保存到: $LOG_FILE"