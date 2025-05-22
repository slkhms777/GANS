#!/bin/bash

# 设置Python模块搜索路径
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 创建日志和可视化目录
mkdir -p Log/terminal
mkdir -p Log/loss_cgan_cifar10_enhanced
mkdir -p visual/CGAN_CIFAR10/progress
mkdir -p checkpoints/CGAN_CIFAR10_ENHANCED

# 获取当前时间作为日志文件名的一部分
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="Log/terminal/cgan_cifar10_enhanced_${TIMESTAMP}.log"

# 打印GPU信息
echo "=== GPU信息 ===" | tee -a $LOG_FILE
nvidia-smi | tee -a $LOG_FILE
echo "===============" | tee -a $LOG_FILE

echo "开始训练 CGAN-CIFAR10 增强模型..." | tee -a $LOG_FILE
echo "使用双3090 GPU并行训练" | tee -a $LOG_FILE
echo "日志将保存到: $LOG_FILE" | tee -a $LOG_FILE

# 记录开始时间
START_TIME=$(date +%s)

# 运行训练脚本
python GAN/CGAN-cifar10-enhanced.py | tee -a $LOG_FILE

# 记录结束时间并计算总训练时间
END_TIME=$(date +%s)
TRAINING_TIME=$((END_TIME - START_TIME))
HOURS=$((TRAINING_TIME / 3600))
MINUTES=$(( (TRAINING_TIME % 3600) / 60 ))
SECONDS=$((TRAINING_TIME % 60))

echo "训练完成！总训练时间: ${HOURS}时 ${MINUTES}分 ${SECONDS}秒" | tee -a $LOG_FILE
echo "查看生成的图片: ./visual/CGAN_CIFAR10/" | tee -a $LOG_FILE
echo "查看训练进度图片: ./visual/CGAN_CIFAR10/progress/" | tee -a $LOG_FILE
echo "查看损失曲线: ./Log/loss_cgan_cifar10_enhanced/" | tee -a $LOG_FILE
echo "模型检查点保存在: ./checkpoints/CGAN_CIFAR10_ENHANCED/" | tee -a $LOG_FILE