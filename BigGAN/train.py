import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import utils as vutils
import numpy as np
from utils import save_checkpoint, sample_5x5_images, save_and_plot_evaluation_scores, FIDEvaluator, ISEvaluator
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

def hinge_g_loss(fake_scores):
    """
    生成器的 Hinge Loss
    目标：最大化 D(G(z)) - 等价于最小化 -D(G(z))
    """
    return -torch.mean(fake_scores)

def hinge_d_loss(real_scores, fake_scores):
    """
    判别器的 Hinge Loss  
    使用 hinge loss 形式：max(0, 1-D(real)) + max(0, 1+D(fake))
    """
    real_loss = torch.mean(F.relu(1.0 - real_scores))
    fake_loss = torch.mean(F.relu(1.0 + fake_scores))
    return real_loss + fake_loss


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
    num_gpus=1,
):
    print("开始训练 MiniBigGAN...")
    
    # 获取原始模型用于保存检查点
    generator_module = generator.module if isinstance(generator, nn.DataParallel) else generator
    discriminator_module = discriminator.module if isinstance(discriminator, nn.DataParallel) else discriminator
    ema_generator_module = ema_generator.module if isinstance(ema_generator, nn.DataParallel) else ema_generator
    
    # 初始化评估器
    is_evaluator = ISEvaluator(device=device)
    fid_evaluator = FIDEvaluator(device=device)
    
    # 准备真实样本用于FID计算
    print("准备真实样本用于FID评估...")
    from utils import get_real_samples_for_evaluation
    real_samples = get_real_samples_for_evaluation(dataloader, num_samples=5000, device=device)
    
    optimizer_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.9))

    # 初始化EMA模型
    ema_generator_module.eval()  

    total_steps = num_epochs * len(dataloader)
    warmup_steps = 10000 
    lr_lambda_func = get_biggan_lr_lambda(warmup_steps, total_steps)
    scheduler_g = LambdaLR(optimizer_g, lr_lambda_func)
    scheduler_d = LambdaLR(optimizer_d, lr_lambda_func)

    # 用于存储每个epoch的平均损失
    epoch_avg_d_hinge_losses = []
    epoch_avg_g_hinge_losses = []
    epoch_avg_ortho_losses = []

    # 初始化评估记录
    fid_scores = []
    is_scores = []
    ema_fid_scores = []
    ema_is_scores = []
    evaluation_epochs = []
    
    evaluation_interval = 50  # 每50个epoch评估一次

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

            # ---------- 训练判别器 ----------
            discriminator.zero_grad()

            z_dim = generator_module.z_dim
            z = torch.randn(images.size(0), z_dim, device=device)
            fake_images_detached = generator(z, labels).detach()

            real_scores = discriminator(images, labels)
            fake_scores_d = discriminator(fake_images_detached, labels)

            d_loss_hinge = hinge_d_loss(real_scores, fake_scores_d)
            d_loss_ortho = orthogonal_regularization(discriminator)
            d_loss_total = d_loss_hinge + d_loss_ortho

            d_loss_total.backward()
            optimizer_d.step()

            # ---------- 训练生成器 ----------
            generator.zero_grad()

            # 重新生成噪声和假图像
            z_g = torch.randn(images.size(0), z_dim, device=device)
            fake_images_for_g = generator(z_g, labels) 
            fake_scores_g = discriminator(fake_images_for_g, labels)

            g_loss_hinge = hinge_g_loss(fake_scores_g)
            g_loss_ortho = orthogonal_regularization(generator)
            g_loss_total = g_loss_hinge + g_loss_ortho

            g_loss_total.backward()
            optimizer_g.step()

            # EMA 更新 - 使用原始模型
            update_ema(ema_generator_module, generator_module, alpha=0.999)

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
        
        # 每10个epoch打印GPU使用情况
        if (epoch + 1) % 10 == 0:
            for gpu_id in range(num_gpus):
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"GPU {gpu_id} 内存使用: {allocated:.2f}GB / {reserved:.2f}GB")

        

        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_checkpoint(generator_module, discriminator_module, ema_generator_module, 
                          optimizer_g, optimizer_d, epoch + 1)
            sample_5x5_images(generator_module, ema_generator_module, epoch + 1, 
                             num_classes=num_classes, device=device)

        # 评估
        if (epoch + 1) % evaluation_interval == 0:
            print(f"\n开始评估 Epoch {epoch + 1}...")
            
            # 导入生成样本函数
            from .utils import generate_samples_for_evaluation
            
            # 使用普通生成器生成样本
            fake_samples = generate_samples_for_evaluation(
                generator_module, 
                num_samples=5000, 
                num_classes=num_classes,
                z_dim=generator_module.z_dim,
                device=device
            )
            
            # 使用EMA生成器生成样本
            ema_fake_samples = generate_samples_for_evaluation(
                ema_generator_module,
                num_samples=5000,
                num_classes=num_classes,
                z_dim=ema_generator_module.z_dim,
                device=device
            )
            
            # 计算评分
            try:
                # 计算普通生成器的评分
                fid_score = fid_evaluator.calculate_fid(real_samples.to(device), fake_samples.to(device))
                is_mean, is_std = is_evaluator.calculate_is(fake_samples.to(device))
                
                # 计算EMA生成器的评分
                ema_fid_score = fid_evaluator.calculate_fid(real_samples.to(device), ema_fake_samples.to(device))
                ema_is_mean, ema_is_std = is_evaluator.calculate_is(ema_fake_samples.to(device))
                
                print(f"Epoch {epoch + 1} - Generator: IS={is_mean:.3f}±{is_std:.3f}, FID={fid_score:.3f}")
                print(f"Epoch {epoch + 1} - EMA Generator: IS={ema_is_mean:.3f}±{ema_is_std:.3f}, FID={ema_fid_score:.3f}")
                
                # 记录评分
                evaluation_epochs.append(epoch + 1)
                fid_scores.append(fid_score)
                is_scores.append((is_mean, is_std))
                ema_fid_scores.append(ema_fid_score)
                ema_is_scores.append((ema_is_mean, ema_is_std))
                
            except Exception as e:
                print(f"评估过程出错: {e}")
                print("跳过本次评估，继续训练...")
    
    # 训练结束后保存所有评估结果
    if evaluation_epochs:
        try:
            from .utils import visual_dir
            save_and_plot_evaluation_scores(
                fid_scores, is_scores, ema_fid_scores, ema_is_scores, 
                evaluation_epochs, save_dir=visual_dir
            )
            print("评估结果已保存完成!")
        except Exception as e:
            print(f"保存评估结果时出错: {e}")

    return epoch_avg_g_hinge_losses, epoch_avg_d_hinge_losses, epoch_avg_ortho_losses






