import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import time
from utils import save_checkpoint, sample_5x5_images
from mini_biggan import MiniBigGANGenerator, MiniBigGANDiscriminator



# 正交正则化 ✅
# Hinge损失 ✅
# 参数平均 ✅
# 学习率schedule ✅

def orthogonal_regularization(model, beta=1e-4):
    ortho_loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            # param的形状是 (out_features, in_features)
            W = param
            WT_W = torch.matmul(W.t(), W)
            I = torch.eye(WT_W.size(0), device=W.device, dtype=W.dtype)
            ortho_loss += ((WT_W - I)**2).sum()
    return beta * ortho_loss

def hinge_loss(real_scores, fake_scores):
    d_loss = torch.mean(F.relu(1.0 - real_scores)) + torch.mean(F.relu(1.0 + fake_scores))
    g_loss = -torch.mean(fake_scores)
    return d_loss, g_loss

def update_ema(ema_model, model, alpha=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def get_biggan_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(warmup_steps)
        else:
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_lambda

def train_miniBigGAN(
    generator,
    discriminator,
    ema_generator,
    device,
    dataloader,
    num_epochs,
    num_classes,
):
    print("开始训练 MiniBigGAN...")
    optimizer_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.9))

    # 初始化EMA模型
    # EMA模型一般只用于推理/评估
    ema_generator.eval()  

    total_steps = num_epochs * len(dataloader)
    warmup_steps = 10000
    lr_lambda = get_biggan_lr_lambda(warmup_steps, total_steps)
    scheduler_g = LambdaLR(optimizer_g, lr_lambda)
    scheduler_d = LambdaLR(optimizer_d, lr_lambda)

    global_step = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            labels = labels.to(device)

            # 生成器训练
            z = torch.randn(images.size(0), generator.z_dim, device=device)
            fake_images = generator(z, labels)

            # 判别器训练
            real_scores = discriminator(images, labels)
            fake_scores = discriminator(fake_images.detach(), labels)

            d_loss, g_loss = hinge_loss(real_scores, fake_scores)

            # 更新判别器
            discriminator.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # 重新计算fake_scores用于生成器训练
            fake_scores = discriminator(fake_images, labels)
            d_loss, g_loss = hinge_loss(real_scores, fake_scores)

            # 更新生成器
            generator.zero_grad()
            g_loss.backward()
            
            # 正交正则化（在optimizer.step之前）
            ortho_loss = orthogonal_regularization(generator) + orthogonal_regularization(discriminator)
            ortho_loss.backward()
            
            optimizer_g.step()

            # EMA更新（在生成器更新后）
            update_ema(ema_generator, generator, alpha=0.999)

            # 更新学习率
            scheduler_g.step()
            scheduler_d.step()
            global_step += 1
        
        # 打印信息
        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Orthogonal Loss: {ortho_loss.item():.4f}")
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_checkpoint(generator, discriminator, ema_generator, optimizer_g, optimizer_d, epoch + 1)
            sample_5x5_images(generator, ema_generator, epoch + 1, device=device)  # 添加device参数





