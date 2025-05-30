import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import utils as vutils
import numpy as np
from .utils import save_checkpoint, sample_5x5_images
from mini_biggan import MiniBigGANGenerator, MiniBigGANDiscriminator
from tqdm import tqdm # 确保导入 tqdm



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
    lr_lambda_func = get_biggan_lr_lambda(warmup_steps, total_steps)
    scheduler_g = LambdaLR(optimizer_g, lr_lambda_func)
    scheduler_d = LambdaLR(optimizer_d, lr_lambda_func)

    # 用于存储每个epoch的平均损失
    epoch_avg_d_hinge_losses = []
    epoch_avg_g_hinge_losses = []
    epoch_avg_ortho_losses = []

    global_step = 0
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        # 初始化每个epoch的累积损失
        running_d_hinge_loss = 0.0
        running_g_hinge_loss = 0.0
        running_d_ortho_loss = 0.0
        running_g_ortho_loss = 0.0

        # 使用 tqdm 包装 dataloader 以显示进度条
        for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(device)
            labels = labels.to(device)

            # --- 训练判别器 ---
            discriminator.zero_grad()
            
            z = torch.randn(images.size(0), generator.z_dim, device=device)
            fake_images_detached = generator(z, labels).detach() # 分离计算图

            real_scores = discriminator(images, labels)
            fake_scores_d = discriminator(fake_images_detached, labels)

            d_loss_hinge, _ = hinge_loss(real_scores, fake_scores_d)
            d_loss_ortho = orthogonal_regularization(discriminator)
            d_loss_total = d_loss_hinge + d_loss_ortho
            
            d_loss_total.backward()
            optimizer_d.step()

            # --- 训练生成器 ---
            generator.zero_grad()

            # 不分离计算图
            fake_images_for_g = generator(z, labels) 
            fake_scores_g = discriminator(fake_images_for_g, labels)
            
            _, g_loss_hinge = hinge_loss(real_scores, fake_scores_g)

            g_loss_ortho = orthogonal_regularization(generator)
            g_loss_total = g_loss_hinge + g_loss_ortho

            g_loss_total.backward()
            optimizer_g.step()

            # EMA 更新
            update_ema(ema_generator, generator, alpha=0.999)

            # 更新学习率
            scheduler_g.step()
            scheduler_d.step()
            global_step += 1

            # 累积当前 batch 的损失
            running_d_hinge_loss += d_loss_hinge.item()
            running_g_hinge_loss += g_loss_hinge.item()
            running_d_ortho_loss += d_loss_ortho.item()
            running_g_ortho_loss += g_loss_ortho.item()
        
        # 计算每个epoch的平均损失
        num_batches = len(dataloader)
        avg_d_hinge_loss_epoch = running_d_hinge_loss / num_batches
        avg_g_hinge_loss_epoch = running_g_hinge_loss / num_batches
        avg_d_ortho_loss_epoch = running_d_ortho_loss / num_batches
        avg_g_ortho_loss_epoch = running_g_ortho_loss / num_batches
        avg_total_ortho_loss_epoch = avg_d_ortho_loss_epoch + avg_g_ortho_loss_epoch

        # 每个epoch结束后保存平均损失
        epoch_avg_d_hinge_losses.append(avg_d_hinge_loss_epoch)
        epoch_avg_g_hinge_losses.append(avg_g_hinge_loss_epoch)
        epoch_avg_ortho_losses.append(avg_total_ortho_loss_epoch)

        # 打印每个epoch的平均损失
        print(f"Epoch [{epoch+1}/{num_epochs}] Completed: "
              f"Avg D Hinge Loss: {avg_d_hinge_loss_epoch:.4f}, "
              f"Avg G Hinge Loss: {avg_g_hinge_loss_epoch:.4f}, "
              f"Avg Ortho Loss (D+G): {avg_total_ortho_loss_epoch:.4f}")

        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_checkpoint(generator, discriminator, ema_generator, optimizer_g, optimizer_d, epoch + 1)
            sample_5x5_images(generator, ema_generator, epoch + 1, num_classes=num_classes, device=device)
    
    return epoch_avg_g_hinge_losses, epoch_avg_d_hinge_losses, epoch_avg_ortho_losses






