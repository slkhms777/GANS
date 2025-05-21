import torch
from torch import nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from utils import get_data
import os
from torch.optim.lr_scheduler import StepLR

data_path = './data'

class Generator(nn.Module):
    def __init__(self, img_size, noise_size, num_hiddens, out_channels=1):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.fc1 = nn.Linear(noise_size, num_hiddens)
        self.bn1 = nn.BatchNorm1d(num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.bn2 = nn.BatchNorm1d(num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, img_size * img_size * out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()  # 输出范围[-1,1]

    def forward(self, noise):
        x = self.relu(self.bn1(self.fc1(noise)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, self.out_channels, self.img_size, self.img_size)
        return x

class Critic(nn.Module):  # 在WGAN中将Discriminator改名为Critic
    def __init__(self, img_size, num_hiddens, in_channels=1):
        super().__init__()
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(img_size * img_size * in_channels, num_hiddens)
        # WGAN中可以不使用dropout，因为权重裁剪可以限制模型复杂度
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, 1)
        self.relu = nn.LeakyReLU(0.2)
        # 注意：WGAN中Critic不使用sigmoid激活，直接输出实数

    def forward(self, image):
        x = self.flt(image)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 直接输出实数，不用sigmoid
        return x

def train_wgan(generator, critic, dataloader, noise_size, num_epochs=10, device='cuda', 
               checkpoint_dir='./checkpoints', n_critic=5, clip_value=0.01):
    """
    WGAN训练函数
    n_critic: 每训练一次生成器，训练批评家的次数
    clip_value: 权重裁剪范围[-clip_value, clip_value]
    """
    generator.to(device)
    critic.to(device)
    generator.train()
    critic.train()
    
    # 保存原始模型，便于后续保存权重
    if isinstance(generator, nn.DataParallel):
        generator_module = generator.module
        critic_module = critic.module
    else:
        generator_module = generator
        critic_module = critic
    
    # WGAN通常使用RMSprop优化器
    optimizer_g = optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_c = optim.RMSprop(critic.parameters(), lr=0.00005)

    # 学习率调度器
    scheduler_g = StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_c = StepLR(optimizer_c, step_size=50, gamma=0.5)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    g_losses = []
    c_losses = []
    wasserstein_distances = []
    
    for epoch in range(num_epochs):
        epoch_g_losses = []
        epoch_c_losses = []
        epoch_wasserstein = []
        
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # 训练批评家
            for _ in range(n_critic):
                noise = torch.randn(batch_size, noise_size, device=device)
                fake_imgs = generator(noise)
                
                # 批评家对真实图像和生成图像的评分
                critic_real = critic(real_imgs)
                critic_fake = critic(fake_imgs.detach())
                
                # Wasserstein距离 = 真实图像评分的均值 - 生成图像评分的均值
                # 批评家希望最大化这个距离，所以是最大化目标（等价于最小化其负值）
                loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake))
                
                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()
                
                # 权重裁剪(重要的WGAN特性)
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            
            # 训练生成器
            noise = torch.randn(batch_size, noise_size, device=device)
            fake_imgs = generator(noise)
            critic_fake = critic(fake_imgs)
            
            # 生成器希望批评家给生成图像高分，所以最大化批评家对生成图像的评分
            loss_g = -torch.mean(critic_fake)
            
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            # 记录每个batch的损失和Wasserstein距离
            with torch.no_grad():
                wasserstein_dist = torch.mean(critic_real) - torch.mean(critic_fake)
                epoch_wasserstein.append(wasserstein_dist.item())
                epoch_c_losses.append(loss_c.item())
                epoch_g_losses.append(loss_g.item())

        avg_c_loss = sum(epoch_c_losses) / len(epoch_c_losses)
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        avg_wasserstein = sum(epoch_wasserstein) / len(epoch_wasserstein)
        
        c_losses.append(avg_c_loss)
        g_losses.append(avg_g_loss)
        wasserstein_distances.append(avg_wasserstein)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss C: {avg_c_loss:.4f}, Loss G: {avg_g_loss:.4f}, "
              f"Wasserstein: {avg_wasserstein:.4f}")

        # 保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save(generator_module.state_dict(), 
                      os.path.join(checkpoint_dir, f'generator_epoch{epoch+1}.pth'))
            torch.save(critic_module.state_dict(), 
                      os.path.join(checkpoint_dir, f'critic_epoch{epoch+1}.pth'))

        scheduler_g.step()
        scheduler_c.step()

        if epoch % 10 == 0:
            # 打印GPU使用情况
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} 内存使用: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB")
    
    return g_losses, c_losses, wasserstein_distances

def save_wgan_losses(g_losses, c_losses, wasserstein_distances, num_epochs):
    """
    保存WGAN损失记录为文本文件和图表
    """
    os.makedirs('./Log/loss_wgan', exist_ok=True)
    
    # 保存为txt文件
    with open('./Log/loss_wgan/wgan_losses.txt', 'w') as f:
        f.write('Epoch,Generator_Loss,Critic_Loss,Wasserstein_Distance\n')
        for i in range(len(g_losses)):
            f.write(f'{i+1},{g_losses[i]},{c_losses[i]},{wasserstein_distances[i]}\n')
    
    # 保存为CSV文件 (可选)
    try:
        import pandas as pd
        df = pd.DataFrame({
            'Epoch': list(range(1, num_epochs+1)),
            'Generator_Loss': g_losses,
            'Critic_Loss': c_losses,
            'Wasserstein_Distance': wasserstein_distances
        })
        df.to_csv('./Log/loss_wgan/wgan_losses.csv', index=False)
    except ImportError:
        print("pandas not installed, skipping CSV export")
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), g_losses, label='Generator')
    plt.plot(range(1, num_epochs+1), c_losses, label='Critic')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), wasserstein_distances, label='Wasserstein Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.legend()
    plt.title('Wasserstein Distance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./Log/loss_wgan/wgan_loss_curve.png')
    plt.close()

def generate_samples(generator, noise_size, num_samples=100, device='cuda', save_path=None):
    generator.to(device)
    generator.eval()
    
    # 保证使用原始模型进行生成
    if isinstance(generator, nn.DataParallel):
        generator_module = generator.module
    else:
        generator_module = generator
    
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_size, device=device)
        # 使用原始模型生成图像
        fake_imgs = generator_module(noise)
    
    # 归一化到[0,1]，适配matplotlib显示
    fake_imgs = (fake_imgs - fake_imgs.min()) / (fake_imgs.max() - fake_imgs.min() + 1e-8)
    grid = vutils.make_grid(fake_imgs, nrow=10, padding=2, normalize=False)
    plt.figure(figsize=(10,10))
    npimg = grid.cpu().numpy()
    if npimg.shape[0] == 1 or npimg.shape[0] == 3:
        # (C,H,W) -> (H,W,C)
        npimg = np.transpose(npimg, (1,2,0))
    plt.axis('off')
    plt.imshow(npimg.squeeze(), cmap='gray' if fake_imgs.shape[1]==1 else None)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭当前图形而不显示
    return fake_imgs.cpu()


if __name__ == '__main__':
    # 检测可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"使用 {num_gpus} 张GPU进行训练")

    # 增大batch_size以充分利用多卡
    batch_size = 256  # 原来是128，有两张卡可以翻倍

    noise_size = 64
    num_hiddens = 512
    img_size = 28
    num_epochs = 200
    n_critic = 5  # 批评家训练次数/生成器训练次数
    clip_value = 0.01  # 权重裁剪范围
    shape = (batch_size, noise_size)
    checkpoint_dir='./checkpoints/WGAN'
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')

    generator = Generator(img_size=img_size, noise_size=noise_size, num_hiddens=num_hiddens, out_channels=1)
    critic = Critic(img_size=img_size, num_hiddens=num_hiddens, in_channels=1)  # 注意这里名称改为critic

    # 使用DataParallel包装模型
    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        critic = nn.DataParallel(critic)

    train_loader, test_loader = get_data.get_train_pics(path=data_path, dataset_name='mnist')

    g_losses, c_losses, wasserstein_distances = train_wgan(
        generator, critic, train_loader, noise_size=noise_size, 
        num_epochs=num_epochs, device=device, n_critic=n_critic, clip_value=clip_value,checkpoint_dir=checkpoint_dir
    )
    
    os.makedirs('./visual/WGAN', exist_ok=True)
    
    for epoch in range(20, num_epochs + 1, 20):
        checkpoint_path = os.path.join(checkpoint_dir, f'generator_epoch{epoch}.pth')
        if os.path.exists(checkpoint_path):
            # 如果是DataParallel，需要加载到module
            if isinstance(generator, nn.DataParallel):
                generator.module.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
            generator.eval()
            print(f"生成 epoch {epoch} 的样本...")
            generate_samples(generator=generator, noise_size=noise_size, 
                           num_samples=100, device=device, 
                           save_path=f'./visual/WGAN/wgan_sample_epoch_{epoch}.png')
            
    save_wgan_losses(g_losses, c_losses, wasserstein_distances, num_epochs)