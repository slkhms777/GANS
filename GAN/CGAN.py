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
    def __init__(self, img_size, noise_size, num_classes, num_hiddens, out_channels=1):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        
        # 噪声和标签的处理层
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # 注意：噪声和嵌入后的标签将拼接在一起
        self.fc1 = nn.Linear(noise_size + num_classes, num_hiddens)
        self.bn1 = nn.BatchNorm1d(num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.bn2 = nn.BatchNorm1d(num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, img_size * img_size * out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()  # 输出范围[-1,1]

    def forward(self, noise, labels):
        # 处理标签
        label_embedding = self.label_embedding(labels)
        
        # 将噪声和标签拼接
        x = torch.cat([noise, label_embedding], dim=1)
        
        # 前向传播
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, self.out_channels, self.img_size, self.img_size)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size, num_classes, num_hiddens, in_channels=1):
        super().__init__()
        self.img_size = img_size
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # 图像处理层
        self.flt = nn.Flatten()
        
        # 拼接后的特征维度 = 图像维度 + 标签嵌入维度
        self.fc1 = nn.Linear(img_size * img_size * in_channels + num_classes, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, image, labels):
        # 处理图像
        img_flat = self.flt(image)
        
        # 处理标签
        label_embedding = self.label_embedding(labels)
        
        # 拼接图像和标签
        x = torch.cat([img_flat, label_embedding], dim=1)
        
        # 前向传播
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_cgan(generator, discriminator, dataloader, noise_size, num_classes, num_epochs=10, device='cuda', checkpoint_dir='./checkpoints'):
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
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 每隔50个epoch，学习率乘以0.5
    scheduler_g = StepLR(optimizer_g, step_size=50, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, step_size=50, gamma=0.5)

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    g_losses = []
    d_losses = []
    
    for epoch in range(num_epochs):
        epoch_g_losses = []
        epoch_d_losses = []
        
        for real_imgs, real_labels in dataloader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_imgs.size(0)
            
            # 创建标签
            real_target = torch.ones(batch_size, 1, device=device)
            fake_target = torch.zeros(batch_size, 1, device=device)

            # 训练判别器
            # 1.1 用真实图像训练
            discriminator.zero_grad()
            outputs_real = discriminator(real_imgs, real_labels)
            loss_real = criterion(outputs_real, real_target)
            
            # 1.2 用生成的假图像训练
            noise = torch.randn(batch_size, noise_size, device=device)
            
            # 随机生成标签 (可选：也可以使用真实标签)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            
            fake_imgs = generator(noise, fake_labels)
            outputs_fake = discriminator(fake_imgs.detach(), fake_labels)
            loss_fake = criterion(outputs_fake, fake_target)
            
            # 总损失
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # 训练生成器
            generator.zero_grad()
            
            # 生成新的假图像
            noise = torch.randn(batch_size, noise_size, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_imgs = generator(noise, fake_labels)
            
            # 判别器尝试对这些假图像进行分类
            outputs = discriminator(fake_imgs, fake_labels)
            
            # 生成器希望判别器认为生成图片为真
            loss_g = criterion(outputs, real_target)
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
    os.makedirs('./Log/loss_cgan', exist_ok=True)
    
    # 保存为txt文件
    with open('./Log/loss_cgan/cgan_losses.txt', 'w') as f:
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
        df.to_csv('./Log/loss_cgan/cgan_losses.csv', index=False)
    except ImportError:
        print("pandas not installed, skipping CSV export")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), g_losses, label='Generator')
    plt.plot(range(1, num_epochs+1), d_losses, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CGAN Training Loss')
    plt.grid(True)
    plt.savefig('./Log/loss_cgan/cgan_loss_curve.png')
    plt.close()

def generate_samples_by_class(generator, noise_size, num_classes, device='cuda', save_path=None):
    """
    按类别生成样本，每个类别生成10个样本
    """
    generator.to(device)
    generator.eval()
    
    # 保证使用原始模型进行生成
    if isinstance(generator, nn.DataParallel):
        generator_module = generator.module
    else:
        generator_module = generator
    
    nrow = 10  # 每行10个样本
    ncol = num_classes  # 每列代表一个类别
    
    with torch.no_grad():
        # 为每个类别生成10个样本
        all_samples = []
        for label in range(num_classes):
            # 为当前类别生成10个样本
            noise = torch.randn(nrow, noise_size, device=device)
            labels = torch.full((nrow,), label, dtype=torch.long, device=device)
            fake_imgs = generator_module(noise, labels)
            all_samples.append(fake_imgs)
        
        # 合并所有类别的样本
        all_samples = torch.cat(all_samples, dim=0)
    
    # 归一化到[0,1]，适配matplotlib显示
    all_samples = (all_samples - all_samples.min()) / (all_samples.max() - all_samples.min() + 1e-8)
    
    # 制作网格
    grid = vutils.make_grid(all_samples, nrow=nrow, padding=2, normalize=False)
    
    # 使用matplotlib显示和保存
    plt.figure(figsize=(15, 15))
    npimg = grid.cpu().numpy()
    if npimg.shape[0] == 1:  # 单通道图像
        npimg = np.transpose(npimg, (1, 2, 0)).squeeze()
        plt.imshow(npimg, cmap='gray')
    else:  # 彩色图像
        npimg = np.transpose(npimg, (1, 2, 0))
        plt.imshow(npimg)
        
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"已保存生成样本到: {save_path}")
    plt.close()
    return all_samples.cpu()


if __name__ == '__main__':
    # 检测可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"使用 {num_gpus} 张GPU进行训练")

    # 增大batch_size以充分利用多卡
    batch_size = 256
    noise_size = 64
    num_hiddens = 512
    img_size = 28
    num_classes = 10  # MNIST有10个类别
    num_epochs = 200
    checkpoint_dir='./checkpoints/CGAN'
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')

    generator = Generator(img_size=img_size, noise_size=noise_size, num_classes=num_classes, 
                         num_hiddens=num_hiddens, out_channels=1)
    discriminator = Discriminator(img_size=img_size, num_classes=num_classes, 
                                 num_hiddens=num_hiddens, in_channels=1)

    # 使用DataParallel包装模型
    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    train_loader, test_loader = get_data.get_train_pics(path=data_path, dataset_name='mnist')

    g_losses, d_losses = train_cgan(generator, discriminator, train_loader, 
                                   noise_size=noise_size, num_classes=num_classes, 
                                   num_epochs=num_epochs, device=device, 
                                   checkpoint_dir=checkpoint_dir)
    
    os.makedirs('./visual/CGAN', exist_ok=True)
    
    # 加载模型并生成样本
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
            generate_samples_by_class(generator=generator, noise_size=noise_size, 
                                    num_classes=num_classes, device=device, 
                                    save_path=f'./visual/CGAN/cgan_sample_epoch_{epoch}.png')
            
    save_losses(g_losses, d_losses, num_epochs)

