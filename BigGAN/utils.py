import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import csv
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import linalg
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2

# 全局路径配置

data_dir = './data'
log_dir = './logs'
checkpoints_dir = './checkpoints'
visual_dir = './visual'


def get_animalfaces_dataloader(batch_size=64, image_size=128, num_workers=8, root_dir=None):
    """
    加载AnimalFaces数据集（cat/dog/wild）为ImageFolder格式的DataLoader
    """
    if root_dir is None:
        root_dir = os.path.join(data_dir, 'AnimalFaces', 'train')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def save_one_batch_images(dataloader, save_path='batch_sample.png', class_names=None):
    """保存一个batch的图片和标签"""
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    images, labels = next(iter(dataloader))
    # 反归一化到[0,1]
    images = images * 0.5 + 0.5

    grid = vutils.make_grid(images, nrow=8, padding=2)
    npimg = grid.cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # 可选：在图片上显示标签
    if class_names is not None:
        for i in range(images.size(0)):
            plt.text((i % 8) * images.size(2) + 5, (i // 8) * images.size(3) + 15,
                     class_names[labels[i]], color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_checkpoint(generator, discriminator, ema_generator, g_optimizer, d_optimizer, epoch, checkpoint_dir=checkpoints_dir):
    """保存生成器和判别器及其优化器的状态"""
    state = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'ema_generator_state_dict': ema_generator.state_dict()
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(generator, discriminator, ema_generator, g_optimizer, d_optimizer, epoch, checkpoint_dir=checkpoints_dir):
    """加载生成器和判别器及其优化器的状态"""
    load_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    state = torch.load(load_path)
    generator.load_state_dict(state['generator_state_dict'])
    discriminator.load_state_dict(state['discriminator_state_dict'])
    g_optimizer.load_state_dict(state['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(state['d_optimizer_state_dict'])
    ema_generator.load_state_dict(state['ema_generator_state_dict'])
    print(f"Checkpoint loaded from {load_path}")


def sample_5x5_images(generator, ema_generator, epoch, num_classes=3, z_dim=128, device='cpu', visual_dir=visual_dir):
    """
    针对 BigGAN 和 AnimalFaces 数据集，生成一个5x5的图片样本
    生成一个5x5的图片样本，前两个类各生成5张，最后一个类生成15张
    """

    y = torch.tensor([0]*5 + [1]*5 + [2]*15, device=device)
    z = torch.randn(25, z_dim, device=device)

    # 处理可能的DataParallel包装
    if hasattr(generator, 'module'):
        generator = generator.module
    if hasattr(ema_generator, 'module'):
        ema_generator = ema_generator.module

    generator.eval()  # 添加eval模式
    ema_generator.eval()  # 添加eval模式

    # 使用普通生成器
    with torch.no_grad():
        fake_imgs = generator(z, y).cpu() * 0.5 + 0.5  # 反归一化到[0,1]

    # 保存到 visual_dir/on_training/generator
    os.makedirs(os.path.join(visual_dir, "on_training", "generator"), exist_ok=True)
    save_path = os.path.join(visual_dir, "on_training", "generator", f"sample_5x5_epoch{epoch}.png")
    vutils.save_image(fake_imgs, save_path, nrow=5)

    # 使用EMA生成器
    with torch.no_grad():
        ema_fake_imgs = ema_generator(z, y).cpu() * 0.5 + 0.5  # 反归一化到[0,1]
    # 保存到 visual_dir/on_training/ema_generator
    os.makedirs(os.path.join(visual_dir, "on_training", "ema_generator"), exist_ok=True)
    ema_save_path = os.path.join(visual_dir, "on_training", "ema_generator", f"sample_5x5_epoch{epoch}.png")
    vutils.save_image(ema_fake_imgs, ema_save_path, nrow=5)

    generator.train()  # 恢复train模式
    # ema_generator保持eval模式

def save_and_plot_losses(g_hinge_losses, d_hinge_losses, ortho_losses, save_dir=visual_dir):
    """
    保存损失到CSV文件并绘制损失曲线图。

    参数:
    g_hinge_losses (list): 每个epoch的生成器Hinge损失列表。
    d_hinge_losses (list): 每个epoch的判别器Hinge损失列表。
    ortho_losses (list): 每个epoch的总正交正则化损失列表。
    save_dir (str): 保存CSV和图像的目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_epochs = len(g_hinge_losses)
    epochs = list(range(1, num_epochs + 1))

    # 1. 保存损失到CSV文件
    csv_file_path = os.path.join(save_dir, 'training_losses.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Generator_Hinge_Loss', 'Discriminator_Hinge_Loss', 'Total_Orthogonal_Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_epochs):
            writer.writerow({
                'Epoch': epochs[i],
                'Generator_Hinge_Loss': g_hinge_losses[i],
                'Discriminator_Hinge_Loss': d_hinge_losses[i],
                'Total_Orthogonal_Loss': ortho_losses[i]
            })
    print(f"损失数据已保存到: {csv_file_path}")

    # 2. 绘制并保存损失曲线图
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, g_hinge_losses, label='Generator Hinge Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Hinge Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, d_hinge_losses, label='Discriminator Hinge Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Hinge Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(epochs, ortho_losses, label='Total Orthogonal Loss (D+G)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Orthogonal Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_file_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(plot_file_path)
    plt.close() # 关闭图像，防止在Jupyter等环境中重复显示
    print(f"损失曲线图已保存到: {plot_file_path}")



class ISEvaluator:
    """Inception Score评估器"""
    def __init__(self, device='cuda', splits=10):
        self.device = device
        self.splits = splits
        
        # 加载预训练的Inception模型
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Linear(self.inception_model.fc.in_features, 1000)  # 保持1000类输出
        self.inception_model.eval()
        self.inception_model.to(device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_inception_probs(self, images):
        """获取Inception预测概率"""
        # images: [N, C, H, W], 范围[0, 1]
        if images.shape[1] == 3 and images.shape[2] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 标准化
        images = self.transform(images)
        
        with torch.no_grad():
            logits = self.inception_model(images)
            probs = F.softmax(logits, dim=1)
        
        return probs
    
    def calculate_is(self, images):
        """计算Inception Score"""
        # images: [N, C, H, W], 范围[0, 1]
        N = images.shape[0]
        
        # 获取预测概率
        probs = self.get_inception_probs(images)
        
        # 分割计算
        split_size = N // self.splits
        scores = []
        
        for i in range(self.splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < self.splits - 1 else N
            
            split_probs = probs[start_idx:end_idx]
            
            # 计算KL散度
            marginal = torch.mean(split_probs, dim=0, keepdim=True)
            kl_div = torch.sum(split_probs * (torch.log(split_probs + 1e-8) - torch.log(marginal + 1e-8)), dim=1)
            scores.append(torch.exp(torch.mean(kl_div)).cpu().item())
        
        return np.mean(scores), np.std(scores)
    

class FIDEvaluator:
    """FID Score评估器"""
    def __init__(self, device='cuda'):
        self.device = device
        
        # 初始化Inception模型用于特征提取
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx]).to(device)
        self.inception_model.eval()
    
    def get_inception_features(self, images):
        """提取Inception特征"""
        # images: [N, C, H, W], 范围[0, 1]
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 转换到[-1, 1]范围
        images = images * 2.0 - 1.0
        
        with torch.no_grad():
            features = self.inception_model(images)[0]
            features = features.squeeze(-1).squeeze(-1)
        
        return features.cpu().numpy()
    
    def calculate_fid(self, real_images, fake_images):
        """计算FID分数"""
        # 提取特征
        real_features = self.get_inception_features(real_images)
        fake_features = self.get_inception_features(fake_images)
        
        # 计算统计量
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
        
        # 计算FID
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
        return fid
    
def save_and_plot_evaluation_scores(fid_scores, is_scores, ema_fid_scores, ema_is_scores, 
                                   evaluation_epochs, save_dir=visual_dir):
    """
    保存FID和IS评分到CSV文件并绘制评分曲线图。

    参数:
    fid_scores (list): 每个评估epoch的FID分数列表
    is_scores (list): 每个评估epoch的IS分数列表 (mean, std)
    ema_fid_scores (list): 每个评估epoch的EMA FID分数列表
    ema_is_scores (list): 每个评估epoch的EMA IS分数列表 (mean, std)
    evaluation_epochs (list): 进行评估的epoch列表
    save_dir (str): 保存CSV和图像的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 确保所有列表长度一致
    assert len(fid_scores) == len(is_scores) == len(ema_fid_scores) == len(ema_is_scores) == len(evaluation_epochs), \
        "所有评分列表长度必须一致"

    # 1. 保存评分到CSV文件
    csv_file_path = os.path.join(save_dir, 'evaluation_scores.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'FID', 'IS_Mean', 'IS_Std', 'EMA_FID', 'EMA_IS_Mean', 'EMA_IS_Std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(evaluation_epochs)):
            writer.writerow({
                'Epoch': evaluation_epochs[i],
                'FID': fid_scores[i],
                'IS_Mean': is_scores[i][0],  # IS分数的均值
                'IS_Std': is_scores[i][1],   # IS分数的标准差
                'EMA_FID': ema_fid_scores[i],
                'EMA_IS_Mean': ema_is_scores[i][0],
                'EMA_IS_Std': ema_is_scores[i][1]
            })
    print(f"评估分数已保存到: {csv_file_path}")

    # 2. 绘制并保存评分曲线图
    plt.figure(figsize=(15, 10))

    # 提取IS均值和标准差
    is_means = [score[0] for score in is_scores]
    is_stds = [score[1] for score in is_scores]
    ema_is_means = [score[0] for score in ema_is_scores]
    ema_is_stds = [score[1] for score in ema_is_scores]

    # FID分数图 (越低越好)
    plt.subplot(2, 2, 1)
    plt.plot(evaluation_epochs, fid_scores, 'bo-', label='Generator FID', linewidth=2, markersize=6)
    plt.plot(evaluation_epochs, ema_fid_scores, 'ro-', label='EMA Generator FID', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title('FID Scores Over Training (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # IS分数图 (越高越好)
    plt.subplot(2, 2, 2)
    plt.errorbar(evaluation_epochs, is_means, yerr=is_stds, fmt='bo-', 
                label='Generator IS', linewidth=2, markersize=6, capsize=5)
    plt.errorbar(evaluation_epochs, ema_is_means, yerr=ema_is_stds, fmt='ro-', 
                label='EMA Generator IS', linewidth=2, markersize=6, capsize=5)
    plt.xlabel('Epoch')
    plt.ylabel('IS Score')
    plt.title('IS Scores Over Training (Higher is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 对比图1: FID对比
    plt.subplot(2, 2, 3)
    width = 0.35
    x = np.arange(len(evaluation_epochs))
    plt.bar(x - width/2, fid_scores, width, label='Generator FID', alpha=0.8)
    plt.bar(x + width/2, ema_fid_scores, width, label='EMA Generator FID', alpha=0.8)
    plt.xlabel('Evaluation Points')
    plt.ylabel('FID Score')
    plt.title('FID Comparison: Generator vs EMA Generator')
    plt.xticks(x, [f'E{ep}' for ep in evaluation_epochs])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 对比图2: IS对比 (只显示均值)
    plt.subplot(2, 2, 4)
    plt.bar(x - width/2, is_means, width, label='Generator IS', alpha=0.8)
    plt.bar(x + width/2, ema_is_means, width, label='EMA Generator IS', alpha=0.8)
    plt.xlabel('Evaluation Points')
    plt.ylabel('IS Score')
    plt.title('IS Comparison: Generator vs EMA Generator')
    plt.xticks(x, [f'E{ep}' for ep in evaluation_epochs])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file_path = os.path.join(save_dir, 'evaluation_curves.png')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"评估曲线图已保存到: {plot_file_path}")

    # 3. 生成评估总结报告
    summary_file_path = os.path.join(save_dir, 'evaluation_summary.txt')
    with open(summary_file_path, 'w') as f:
        f.write("BigGAN 评估总结报告\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"评估间隔: 每 {evaluation_epochs[1] - evaluation_epochs[0]} 个epoch\n")
        f.write(f"评估轮次: {len(evaluation_epochs)} 次\n")
        f.write(f"评估epoch: {evaluation_epochs}\n\n")
        
        # 最佳分数
        best_fid_idx = np.argmin(fid_scores)
        best_ema_fid_idx = np.argmin(ema_fid_scores)
        best_is_idx = np.argmax(is_means)
        best_ema_is_idx = np.argmax(ema_is_means)
        
        f.write("最佳分数:\n")
        f.write("-" * 20 + "\n")
        f.write(f"最佳 FID: {fid_scores[best_fid_idx]:.3f} (Epoch {evaluation_epochs[best_fid_idx]})\n")
        f.write(f"最佳 EMA FID: {ema_fid_scores[best_ema_fid_idx]:.3f} (Epoch {evaluation_epochs[best_ema_fid_idx]})\n")
        f.write(f"最佳 IS: {is_means[best_is_idx]:.3f}±{is_stds[best_is_idx]:.3f} (Epoch {evaluation_epochs[best_is_idx]})\n")
        f.write(f"最佳 EMA IS: {ema_is_means[best_ema_is_idx]:.3f}±{ema_is_stds[best_ema_is_idx]:.3f} (Epoch {evaluation_epochs[best_ema_is_idx]})\n\n")
        
        # 最终分数
        f.write("最终分数:\n")
        f.write("-" * 20 + "\n")
        f.write(f"最终 FID: {fid_scores[-1]:.3f}\n")
        f.write(f"最终 EMA FID: {ema_fid_scores[-1]:.3f}\n")
        f.write(f"最终 IS: {is_means[-1]:.3f}±{is_stds[-1]:.3f}\n")
        f.write(f"最终 EMA IS: {ema_is_means[-1]:.3f}±{ema_is_stds[-1]:.3f}\n\n")
        
        # 改善趋势
        fid_improvement = fid_scores[0] - fid_scores[-1]
        ema_fid_improvement = ema_fid_scores[0] - ema_fid_scores[-1]
        is_improvement = is_means[-1] - is_means[0]
        ema_is_improvement = ema_is_means[-1] - ema_is_means[0]
        
        f.write("训练改善:\n")
        f.write("-" * 20 + "\n")
        f.write(f"FID 改善: {fid_improvement:.3f} ({'提升' if fid_improvement > 0 else '下降'})\n")
        f.write(f"EMA FID 改善: {ema_fid_improvement:.3f} ({'提升' if ema_fid_improvement > 0 else '下降'})\n")
        f.write(f"IS 改善: {is_improvement:.3f} ({'提升' if is_improvement > 0 else '下降'})\n")
        f.write(f"EMA IS 改善: {ema_is_improvement:.3f} ({'提升' if ema_is_improvement > 0 else '下降'})\n")

    print(f"评估总结报告已保存到: {summary_file_path}")


def generate_samples_for_evaluation(generator, num_samples, num_classes, z_dim, device, batch_size=100):
    """生成用于评估的样本"""
    generator.eval()
    all_samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # 随机生成噪声和标签
            z = torch.randn(current_batch_size, z_dim, device=device)
            y = torch.randint(0, num_classes, (current_batch_size,), device=device)
            
            # 生成图像
            fake_images = generator(z, y)
            
            # 反归一化到[0, 1]
            fake_images = (fake_images + 1.0) / 2.0
            fake_images = torch.clamp(fake_images, 0, 1)
            
            all_samples.append(fake_images.cpu())
    
    generator.train()  # 恢复训练模式
    return torch.cat(all_samples, dim=0)


def get_real_samples_for_evaluation(dataloader, num_samples, device):
    """从数据集中获取真实样本"""
    all_real_images = []
    count = 0
    
    for images, _ in dataloader:
        if count >= num_samples:
            break
            
        batch_size = images.shape[0]
        take = min(batch_size, num_samples - count)
        
        # 反归一化到[0, 1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0, 1)
        
        all_real_images.append(images[:take])
        count += take
    
    return torch.cat(all_real_images, dim=0)