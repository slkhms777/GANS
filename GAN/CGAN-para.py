print("脚本已开始执行")
import torch
from torch import nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from utils import get_data
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# 导入DDP和AMP相关库
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

data_path = './data'

# 自注意力和模型定义保持不变...
class SelfAttention(nn.Module):
    """自注意力机制模块，用于提升生成图像的全局一致性"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 投影查询、键和值
        proj_query = self.query(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height*width)
        energy = torch.bmm(proj_query, proj_key)  # 批量矩阵乘法
        
        # 注意力图
        attention = self.softmax(energy)
        
        # 输出
        proj_value = self.value(x).view(batch_size, -1, height*width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out

class Generator(nn.Module):
    def __init__(self, img_size=32, noise_size=128, num_classes=10, num_hiddens=512, out_channels=3):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        
        # 标签嵌入层 - 增大嵌入维度
        self.label_embedding = nn.Embedding(num_classes, num_classes * 2)
        
        # 初始特征图尺寸
        self.init_size = img_size // 8  # 4x4 起始特征图
        self.l1 = nn.Sequential(
            nn.Linear(noise_size + num_classes * 2, num_hiddens * 8 * self.init_size * self.init_size),
            nn.BatchNorm1d(num_hiddens * 8 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 转置卷积部分 - 增加层数和特征通道
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(num_hiddens * 8, num_hiddens * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 自注意力模块
            SelfAttention(num_hiddens * 4),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(num_hiddens * 4, num_hiddens * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(num_hiddens * 2, num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最终输出层
            nn.Conv2d(num_hiddens, out_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 处理标签
        label_embedding = self.label_embedding(labels)
        
        # 将噪声和标签拼接
        x = torch.cat([noise, label_embedding], dim=1)
        
        # 前向传播
        x = self.l1(x)
        x = x.view(x.shape[0], -1, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size=32, num_classes=10, num_hiddens=512, in_channels=3):
        super().__init__()
        self.img_size = img_size
        
        # 标签嵌入层 - 增大嵌入维度
        self.label_embedding = nn.Embedding(num_classes, num_classes * 2)
        
        # 图像卷积部分 - 增加层数和特征通道
        self.conv_blocks = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(in_channels, num_hiddens, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(num_hiddens, num_hiddens * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            # 自注意力模块
            SelfAttention(num_hiddens * 2),
            
            # 8x8 -> 4x4
            nn.Conv2d(num_hiddens * 2, num_hiddens * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            
            # 增加一层以提高表达能力
            nn.Conv2d(num_hiddens * 4, num_hiddens * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_hiddens * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
        )
        
        # 展平后卷积特征的大小
        ds_size = img_size // 8  # 4x4
        
        # 分类器部分(连接卷积特征和标签嵌入)
        self.classifier = nn.Sequential(
            nn.Linear(num_hiddens * 8 * ds_size * ds_size + num_classes * 2, num_hiddens * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(num_hiddens * 4, num_hiddens * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(num_hiddens * 2, 1)
        )

    def forward(self, image, labels):
        # 处理图像
        x = self.conv_blocks(image)
        x = x.view(x.shape[0], -1)
        
        # 处理标签
        label_embedding = self.label_embedding(labels)
        
        # 拼接图像特征和标签
        x = torch.cat([x, label_embedding], dim=1)
        
        # 分类器
        x = self.classifier(x)
        return x

def train_cgan(generator, discriminator, dataloader, noise_size, num_classes, num_epochs=10, device='cuda', 
              checkpoint_dir='./checkpoints', local_rank=0, world_size=1):
    """
    使用DDP和AMP的训练函数
    """
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    
    # 获取原始模型（用于保存权重）
    if isinstance(generator, DDP):
        generator_module = generator.module
        discriminator_module = discriminator.module
    else:
        generator_module = generator
        discriminator_module = discriminator
        
    criterion = nn.BCEWithLogitsLoss()
    # 优化器参数调整
    optimizer_g = optim.Adam(generator.parameters(), lr=0.00015, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.00025, betas=(0.5, 0.999))

    # 学习率调度器
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=num_epochs, eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=num_epochs, eta_min=1e-6)

    # 创建混合精度训练的Scaler
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    if local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    g_losses = []
    d_losses = []
    
    for epoch in range(num_epochs):
        # 设置 sampler 的 epoch，以确保洗牌(shuffle)是确定的
        dataloader.sampler.set_epoch(epoch)
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        for real_imgs, real_labels in dataloader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_imgs.size(0)
            
            # 创建标签并添加平滑化和随机噪声
            real_target = torch.ones(batch_size, 1, device=device) * 0.9 + torch.rand(batch_size, 1, device=device) * 0.1
            fake_target = torch.zeros(batch_size, 1, device=device) + torch.rand(batch_size, 1, device=device) * 0.1

            # ---------- 训练判别器 ----------
            discriminator.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                # 用真实图像训练
                outputs_real = discriminator(real_imgs, real_labels)
                loss_real = criterion(outputs_real, real_target)
                
                # 用生成的假图像训练
                noise = torch.randn(batch_size, noise_size, device=device)
                fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
                
                fake_imgs = generator(noise, fake_labels)
                outputs_fake = discriminator(fake_imgs.detach(), fake_labels)
                loss_fake = criterion(outputs_fake, fake_target)
                
                # 总损失
                loss_d = loss_real + loss_fake
            
            # 使用混合精度缩放梯度并更新
            scaler_d.scale(loss_d).backward()
            # 梯度裁剪，防止梯度爆炸
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # ---------- 训练生成器 ----------
            if batch_size % 2 == 0:  # 每2个batch训练一次生成器，平衡训练
                generator.zero_grad()
                
                with autocast():
                    # 生成新的假图像
                    noise = torch.randn(batch_size, noise_size, device=device)
                    fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
                    fake_imgs = generator(noise, fake_labels)
                    
                    # 判别器尝试对这些假图像进行分类
                    outputs = discriminator(fake_imgs, fake_labels)
                    
                    # 生成器希望判别器认为生成图片为真
                    loss_g = criterion(outputs, real_target)
                
                # 使用混合精度缩放梯度并更新
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
                
                # 记录损失
                epoch_g_losses.append(loss_g.item())
            
            # 记录判别器损失
            epoch_d_losses.append(loss_d.item())

        # 收集所有进程上的损失
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses) if epoch_g_losses else 0
        
        # 在多卡训练时确保损失值同步
        if world_size > 1:
            # 创建张量储存平均损失
            avg_d_loss_tensor = torch.tensor(avg_d_loss).to(device)
            avg_g_loss_tensor = torch.tensor(avg_g_loss).to(device)
            
            # 将所有进程中的损失平均
            dist.all_reduce(avg_d_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_g_loss_tensor, op=dist.ReduceOp.SUM)
            
            # 计算所有进程平均值
            avg_d_loss = avg_d_loss_tensor.item() / world_size
            avg_g_loss = avg_g_loss_tensor.item() / world_size
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # 只在主进程(rank 0)打印和保存
        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")

            # 保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save(generator_module.state_dict(), 
                          os.path.join(checkpoint_dir, f'generator_epoch{epoch+1}.pth'))
                torch.save(discriminator_module.state_dict(), 
                          os.path.join(checkpoint_dir, f'discriminator_epoch{epoch+1}.pth'))
                
                # 每10个epoch生成一次样本，便于监控训练进展
                with torch.no_grad():
                    test_noise = torch.randn(100, noise_size, device=device)
                    test_labels = torch.tensor([i for i in range(10) for _ in range(10)], device=device)
                    with autocast():
                        gen_imgs = generator(test_noise, test_labels)
                    os.makedirs('./visual/CGAN_CIFAR10/progress', exist_ok=True)
                    save_image_grid(gen_imgs, f'./visual/CGAN_CIFAR10/progress/epoch_{epoch+1}.png', nrow=10)

            # 在主进程打印显存使用情况
            if (epoch + 1) % 5 == 0 or epoch == 0:
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i} 内存使用: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB / {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f}GB")

        scheduler_g.step()
        scheduler_d.step()
    
    # 同步所有进程，确保训练完成
    if world_size > 1:
        dist.barrier()
    
    return g_losses, d_losses

def save_image_grid(images, path, nrow=8):
    """保存图像网格"""
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, padding=2)
    plt.figure(figsize=(10, 10))
    # 将FP16张量转换为FP32，避免警告
    if images.dtype == torch.float16:
        grid = grid.float()
    npimg = grid.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def save_losses(g_losses, d_losses, num_epochs):
    """保存损失记录为文本文件和图表"""
    os.makedirs('./Log/loss_cgan_cifar10_enhanced', exist_ok=True)
    
    # 保存为txt文件
    with open('./Log/loss_cgan_cifar10_enhanced/cgan_losses.txt', 'w') as f:
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
        df.to_csv('./Log/loss_cgan_cifar10_enhanced/cgan_losses.csv', index=False)
    except ImportError:
        print("pandas not installed, skipping CSV export")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), g_losses, label='Generator')
    plt.plot(range(1, num_epochs+1), d_losses, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CGAN-CIFAR10 Training Loss')
    plt.grid(True)
    plt.savefig('./Log/loss_cgan_cifar10_enhanced/cgan_loss_curve.png')
    plt.close()

def generate_samples_by_class(generator, noise_size, num_classes, device='cuda', save_path=None):
    """按类别生成样本，每个类别生成10个样本"""
    generator.to(device)
    generator.eval()
    
    # 保证使用原始模型进行生成
    if isinstance(generator, (nn.DataParallel, DDP)):
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
            # 使用混合精度生成样本
            with autocast():
                fake_imgs = generator_module(noise, labels)
            # 如果是FP16，转为FP32
            if fake_imgs.dtype == torch.float16:
                fake_imgs = fake_imgs.float()
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
    npimg = np.transpose(npimg, (1, 2, 0))  # CIFAR是彩色图像
    plt.imshow(npimg)
        
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"已保存生成样本到: {save_path}")
    plt.close()
    return all_samples.cpu()

def setup_ddp():
    """设置分布式训练环境"""
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    # 获取世界大小
    world_size = dist.get_world_size()
    
    # 获取本地rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size

def main():
    # 设置DDP环境
    local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # 只在主进程(rank 0)打印信息
    if local_rank == 0:
        print(f"使用 {world_size} 张GPU进行训练")
        print(f"本地进程使用GPU: {local_rank}")

    # 超参数配置 - 针对双3090进行优化
    batch_size = 512   # 减小每个GPU上的batch size，总批量为 batch_size * world_size
    noise_size = 128    # 保持噪声维度不变
    num_hiddens = 512   # 保持基础通道数不变
    img_size = 32       # CIFAR-10是32x32
    num_classes = 10    # CIFAR-10有10个类别
    num_epochs = 120    # 训练轮数
    
    checkpoint_dir='./checkpoints/CGAN_CIFAR10_ENHANCED_DDP'

    # 创建增强版模型
    generator = Generator(img_size=img_size, noise_size=noise_size, num_classes=num_classes, 
                         num_hiddens=num_hiddens, out_channels=3)
    discriminator = Discriminator(img_size=img_size, num_classes=num_classes, 
                                 num_hiddens=num_hiddens, in_channels=3)

    # 打印模型参数量，便于监控（仅在主进程）
    if local_rank == 0:
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"生成器参数量: {count_parameters(generator)/1e6:.2f}M")
        print(f"判别器参数量: {count_parameters(discriminator)/1e6:.2f}M")

    # 将模型移至对应设备并转换为DDP模型
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator = DDP(generator, device_ids=[local_rank], output_device=local_rank)
    discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)
    
    if local_rank == 0:
        print(f"模型已启用DDP并行训练，使用{world_size}张GPU")

    # 获取数据集
    train_dataset = get_data.get_cifar10_dataset(path=data_path)
    
    # 使用DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=local_rank
    )
    
    # 创建DataLoader，使用sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    if local_rank == 0:
        print("开始训练")
        
    # 训练模型
    g_losses, d_losses = train_cgan(
        generator, discriminator, train_loader, 
        noise_size=noise_size, num_classes=num_classes, 
        num_epochs=num_epochs, device=device, 
        checkpoint_dir=checkpoint_dir,
        local_rank=local_rank,
        world_size=world_size
    )
    
    # 只在主进程保存损失曲线
    if local_rank == 0:
        os.makedirs('./Log/loss_cgan_cifar10_enhanced', exist_ok=True)
        save_losses(g_losses, d_losses, num_epochs)
        print("训练完成，已保存损失曲线和模型检查点")

    # 清理DDP资源
    dist.destroy_process_group()

if __name__ == '__main__':
    # 注意：这种方式必须通过torch.distributed.launch或torchrun启动
    main()