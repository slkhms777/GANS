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
        
        # 增加网络深度和宽度
        self.net = nn.Sequential(
            nn.Linear(noise_size, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(num_hiddens, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(num_hiddens, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(num_hiddens, img_size * img_size * out_channels),
            nn.Tanh()
        )

    def forward(self, noise):
        x = self.net(noise)
        x = x.view(-1, self.out_channels, self.img_size, self.img_size)
        return x

class Critic(nn.Module):
    def __init__(self, img_size, num_hiddens, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(img_size * img_size * in_channels, num_hiddens),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(num_hiddens, num_hiddens),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(num_hiddens, num_hiddens),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(num_hiddens, 1)
        )

    def forward(self, image):
        return self.net(image)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算梯度惩罚项"""
    # 在真实样本和生成样本之间随机插值
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    # 插值样本
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # 计算批评家对插值样本的输出
    d_interpolates = critic(interpolates)
    # 创建与d_interpolates形状相同的全1张量，用于计算梯度
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # 计算梯度的范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(generator, critic, dataloader, noise_size, num_epochs=100, device='cuda', 
               checkpoint_dir='./checkpoints', n_critic=5, lambda_gp=10):
    """
    WGAN-GP训练函数
    n_critic: 每训练一次生成器，训练批评家的次数
    lambda_gp: 梯度惩罚的权重
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
    
    # WGAN-GP使用Adam优化器，且学习率一般比WGAN高
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))

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
                
                # 计算梯度惩罚
                gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs.detach(), device)
                
                # Wasserstein距离 + 梯度惩罚
                loss_c = -torch.mean(critic_real) + torch.mean(critic_fake) + lambda_gp * gradient_penalty
                
                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()
                
                # 不再需要权重裁剪
            
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
    保存WGAN-GP损失记录为文本文件和图表
    """
    os.makedirs('./Log/loss_wgan_gp', exist_ok=True)
    
    # 保存为txt文件
    with open('./Log/loss_wgan_gp/wgan_gp_losses.txt', 'w') as f:
        f.write('Epoch,Generator_Loss,Critic_Loss,Wasserstein_Distance\n')
        for i in range(len(g_losses)):
            f.write(f'{i+1},{g_losses[i]},{c_losses[i]},{wasserstein_distances[i]}\n')
    
    # 保存为CSV文件
    try:
        import pandas as pd
        df = pd.DataFrame({
            'Epoch': list(range(1, num_epochs+1)),
            'Generator_Loss': g_losses,
            'Critic_Loss': c_losses,
            'Wasserstein_Distance': wasserstein_distances
        })
        df.to_csv('./Log/loss_wgan_gp/wgan_gp_losses.csv', index=False)
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
    plt.title('WGAN-GP Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), wasserstein_distances, label='Wasserstein Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.legend()
    plt.title('Wasserstein Distance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./Log/loss_wgan_gp/wgan_gp_loss_curve.png')
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

    # 调整超参数
    batch_size = 128  # 对WGAN-GP适当调整batch size
    noise_size = 100  # 增加噪声维度
    num_hiddens = 1024  # 增加隐藏层宽度
    img_size = 28
    num_epochs = 200
    n_critic = 5  # 批评家训练次数/生成器训练次数
    lambda_gp = 10  # 梯度惩罚权重
    checkpoint_dir = './checkpoints/WGAN-GP'  # 修改检查点目录
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')

    generator = Generator(img_size=img_size, noise_size=noise_size, num_hiddens=num_hiddens, out_channels=1)
    critic = Critic(img_size=img_size, num_hiddens=num_hiddens, in_channels=1)

    # 使用DataParallel包装模型
    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        critic = nn.DataParallel(critic)

    train_loader, test_loader = get_data.get_train_pics(path=data_path, dataset_name='mnist')

    g_losses, c_losses, wasserstein_distances = train_wgan_gp(
        generator, critic, train_loader, noise_size=noise_size, 
        num_epochs=num_epochs, device=device, n_critic=n_critic, lambda_gp=lambda_gp,
        checkpoint_dir=checkpoint_dir
    )
    
    os.makedirs('./visual/WGAN-GP', exist_ok=True)  # 修改可视化目录
    
    for epoch in range(20, num_epochs + 1, 20):
        checkpoint_path = os.path.join(checkpoint_dir, f'generator_epoch{epoch}.pth')
        if os.path.exists(checkpoint_path):
            if isinstance(generator, nn.DataParallel):
                generator.module.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
            generator.eval()
            print(f"生成 epoch {epoch} 的样本...")
            generate_samples(generator=generator, noise_size=noise_size, 
                           num_samples=100, device=device, 
                           save_path=f'./visual/WGAN-GP/wgan_gp_sample_epoch_{epoch}.png')
            
    save_wgan_losses(g_losses, c_losses, wasserstein_distances, num_epochs)