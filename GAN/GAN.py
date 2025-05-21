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
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.relu(self.bn1(self.fc1(noise)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, self.out_channels, self.img_size, self.img_size)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size, num_hiddens, in_channels=1):
        super().__init__()
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(img_size * img_size * in_channels, num_hiddens)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(num_hiddens, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, image):
        x = self.flt(image)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_gan(generator, discriminator, dataloader, noise_size, num_epochs=10, device='cuda', checkpoint_dir='./checkpoints'):
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    
    # 保存原始模型，便于后续保存权重
    if isinstance(generator, nn.DataParallel):
        generator_module = generator.module
        discriminator_module = discriminator.module
    else:
        generator_module = generator
        discriminator_module = discriminator
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    # 每隔50个epoch，学习率乘以0.5
    scheduler_g = StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, step_size=50, gamma=0.5)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    g_losses = []
    d_losses = []
    
    for epoch in range(num_epochs):
        epoch_g_losses = []
        epoch_d_losses = []
        
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 训练判别器
            noise = torch.randn(batch_size, noise_size, device=device)
            fake_imgs = generator(noise)
            outputs_real = discriminator(real_imgs)
            outputs_fake = discriminator(fake_imgs.detach())
            loss_d_real = criterion(outputs_real, real_labels)
            loss_d_fake = criterion(outputs_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # 训练生成器
            noise = torch.randn(batch_size, noise_size, device=device)
            fake_imgs = generator(noise)
            outputs = discriminator(fake_imgs)
            loss_g = criterion(outputs, real_labels) # 希望判别器认为生成图片为真

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            # 记录每个batch的损失
            epoch_d_losses.append(loss_d.item())
            epoch_g_losses.append(loss_g.item())

        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save(generator_module.state_dict(), 
                      os.path.join(checkpoint_dir, f'generator_epoch{epoch+1}.pth'))
            torch.save(discriminator_module.state_dict(), 
                      os.path.join(checkpoint_dir, f'discriminator_epoch{epoch+1}.pth'))

        scheduler_g.step()
        scheduler_d.step()

        if epoch % 10 == 0:
            # 打印GPU使用情况
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} 内存使用: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB")
    
    return g_losses, d_losses

def save_losses(g_losses, d_losses, num_epochs):
    """
    保存损失记录为文本文件和图表
    """
    os.makedirs('./Log/loss_gan', exist_ok=True)
    
    # 保存为txt文件
    with open('./Log/loss_gan/gan_losses.txt', 'w') as f:
        f.write('Epoch,Generator_Loss,Discriminator_Loss\n')
        for i in range(len(g_losses)):
            f.write(f'{i+1},{g_losses[i]},{d_losses[i]}\n')
    
    # 保存为CSV文件 (可选)
    try:
        import pandas as pd
        df = pd.DataFrame({
            'Epoch': list(range(1, num_epochs+1)),
            'Generator_Loss': g_losses,
            'Discriminator_Loss': d_losses
        })
        df.to_csv('./Log/loss_gan/gan_losses.csv', index=False)
    except ImportError:
        print("pandas not installed, skipping CSV export")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), g_losses, label='Generator')
    plt.plot(range(1, num_epochs+1), d_losses, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Loss')
    plt.grid(True)
    plt.savefig('./Log/loss_gan/gan_loss_curve.png')
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
    shape = (batch_size, noise_size)
    checkpoint_dir='./checkpoints/GAN'
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')


    generator = Generator(img_size=img_size, noise_size=noise_size, num_hiddens=num_hiddens, out_channels=1)
    discriminator = Discriminator(img_size=img_size, num_hiddens=num_hiddens, in_channels=1)

    # 使用DataParallel包装模型
    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    train_loader, test_loader = get_data.get_train_pics(path=data_path,dataset_name='mnist')

    g_losses, d_losses = train_gan(generator, discriminator, train_loader, noise_size=noise_size, num_epochs=num_epochs,device=device)
    
    os.makedirs('./visual/GAN', exist_ok=True)
    
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
                           num_samples=100, device=device, save_path=f'./visual/GAN/gan_sample_epoch_{epoch}.png')
            
    save_losses(g_losses, d_losses, num_epochs)
