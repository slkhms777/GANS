#!/bin/bash

# 设置Python模块搜索路径
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 创建日志目录
mkdir -p Log/terminal

# 获取当前时间作为日志文件名的一部分
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="Log/terminal/wgan_training_${TIMESTAMP}.log"

echo "开始训练 WGAN 模型..."
echo "日志将保存到: $LOG_FILE"

python GAN/WGAN.py | tee $LOG_FILE

echo "训练完成！"
echo "查看生成的图片: ./visual/WGAN"
echo "查看损失曲线: ./Log/loss_wgan/"