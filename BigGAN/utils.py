import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils


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