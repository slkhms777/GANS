import torch
from torch import nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import cv2
from PIL import Image
from GAN.CGAN import Generator, Discriminator

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

# 确保生成器移动到正确的设备
generator = generator.to(device)

checkpoint_path = './checkpoints/CGAN/generator_epoch100.pth'
if isinstance(generator, nn.DataParallel):
    generator.module.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()

def enhance_image_quality(img_tensor, target_size=128):
    """
    使用超分辨率技术增强图像质量
    """
    # 转换为numpy数组
    img = img_tensor.cpu().numpy()
    
    # 归一化到0-255
    img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)
    
    # 使用双三次插值放大
    enhanced = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    # 应用锐化滤波器
    kernel = np.array([[-0.5, -1, -0.5],
                      [-1, 7, -1],
                      [-0.5, -1, -0.5]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 应用轻微的高斯模糊以减少锯齿
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    
    # 对比度增强
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
    
    # 限制范围
    enhanced = np.clip(enhanced, 0, 255)
    
    return enhanced

def create_high_quality_grid(images, num_step, target_size=128, dpi=300):
    """
    创建高质量的网格可视化
    """
    # 设置matplotlib参数以获得更好的质量
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.5
    
    # 计算图像尺寸
    fig_width = num_step * 2.5
    fig_height = 3
    
    fig, axes = plt.subplots(1, num_step, figsize=(fig_width, fig_height), 
                            facecolor='white', edgecolor='none')
    
    # 如果只有一个子图，确保axes是列表
    if num_step == 1:
        axes = [axes]
    
    enhanced_images = []
    
    for i in range(num_step):
        # 获取图像并增强质量
        img = images[i].squeeze()
        enhanced_img = enhance_image_quality(img, target_size)
        enhanced_images.append(enhanced_img)
        
        # 显示图像
        axes[i].imshow(enhanced_img, cmap='gray', interpolation='bicubic', 
                      vmin=0, vmax=255, aspect='equal')
        
        # 设置标题
        alpha_value = i / (num_step - 1) if num_step > 1 else 0
        axes[i].set_title(f'α={alpha_value:.2f}', fontsize=12, fontweight='bold', pad=10)
        
        # 移除坐标轴
        axes[i].axis('off')
        
        # 设置边框
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    
    # 调整子图间距
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    return fig, enhanced_images

def smooth_interpolation(start_tensor, end_tensor, num_steps, method='cosine'):
    """
    使用更平滑的插值方法
    """
    interpolated = []
    
    for i in range(num_steps):
        if method == 'cosine':
            # 余弦插值，提供更平滑的过渡
            alpha = 0.5 * (1 - np.cos(np.pi * i / (num_steps - 1)))
        elif method == 'sigmoid':
            # S型插值
            x = (i / (num_steps - 1) - 0.5) * 6
            alpha = 1 / (1 + np.exp(-x))
        else:
            # 线性插值
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
        
        interp_tensor = (1 - alpha) * start_tensor + alpha * end_tensor
        interpolated.append(interp_tensor)
    
    return interpolated

def advanced_interpolation_visual(generator, start_label=1, end_label=6, num_step=15, 
                                interpolation_method='cosine', save_individual=True):
    """
    改进的高质量插值函数
    """
    generator.eval()
    
    with torch.no_grad():
        # 生成多组噪声向量并选择最佳的
        num_candidates = 5
        best_images = None
        best_score = float('inf')
        
        for attempt in range(num_candidates):
            # 生成起始和结束的噪声向量
            z1 = torch.randn(1, noise_size, device=device)
            z2 = torch.randn(1, noise_size, device=device)
            
            # 获取起始和结束标签的嵌入向量
            start_label_tensor = torch.tensor([start_label], device=device)
            end_label_tensor = torch.tensor([end_label], device=device)
            
            # 获取标签嵌入
            if isinstance(generator, nn.DataParallel):
                start_embed = generator.module.label_embedding(start_label_tensor)
                end_embed = generator.module.label_embedding(end_label_tensor)
            else:
                start_embed = generator.label_embedding(start_label_tensor)
                end_embed = generator.label_embedding(end_label_tensor)
            
            # 使用平滑插值
            z_interpolated = smooth_interpolation(z1, z2, num_step, interpolation_method)
            embed_interpolated = smooth_interpolation(start_embed, end_embed, num_step, interpolation_method)
            
            interpolated_images = []
            
            for i in range(num_step):
                # 拼接噪声和标签嵌入
                x = torch.cat([z_interpolated[i], embed_interpolated[i]], dim=1)
                
                # 通过生成器的各层
                if isinstance(generator, nn.DataParallel):
                    x = generator.module.fc1(x)
                    x = generator.module.bn1(x)
                    x = torch.relu(x)
                    x = generator.module.fc2(x)
                    x = generator.module.bn2(x)
                    x = torch.relu(x)
                    generated_img = generator.module.fc3(x)
                else:
                    x = generator.fc1(x)
                    x = generator.bn1(x)
                    x = torch.relu(x)
                    x = generator.fc2(x)
                    x = generator.bn2(x)
                    x = torch.relu(x)
                    generated_img = generator.fc3(x)
                
                # 重塑为图像格式
                generated_img = generated_img.view(-1, 1, img_size, img_size)
                generated_img = torch.tanh(generated_img)
                
                interpolated_images.append(generated_img)
            
            # 评估这组图像的质量（这里使用简单的方差作为质量指标）
            quality_score = sum([img.var().item() for img in interpolated_images])
            
            if quality_score < best_score:
                best_score = quality_score
                best_images = interpolated_images
        
        # 使用最佳图像
        interpolated_images = torch.cat(best_images, dim=0)
        
        # 创建高质量可视化
        fig, enhanced_images = create_high_quality_grid(interpolated_images, num_step, 
                                                       target_size=128, dpi=300)
        
        # 设置主标题
        plt.suptitle(f'High-Quality Interpolation: Digit {start_label} → Digit {end_label}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 确保目录存在
        import os
        os.makedirs('./visual/interpolation', exist_ok=True)
        os.makedirs('./visual/interpolation/individual', exist_ok=True)
        
        # 保存高质量主图
        main_filename = f'./visual/interpolation/{start_label}_to_{end_label}_hq.png'
        plt.savefig(main_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        
        # 保存单独的高清图像
        if save_individual:
            for i, enhanced_img in enumerate(enhanced_images):
                alpha_value = i / (num_step - 1) if num_step > 1 else 0
                individual_filename = f'./visual/interpolation/individual/{start_label}_to_{end_label}_step_{i:02d}_alpha_{alpha_value:.2f}.png'
                
                # 使用PIL保存高质量单独图像
                pil_img = Image.fromarray(enhanced_img, mode='L')
                pil_img.save(individual_filename, 'PNG', quality=95, optimize=True)
        
        plt.show()
        
        return interpolated_images, enhanced_images

def create_comparison_visualization(generator, start_label=1, end_label=6):
    """
    创建不同插值方法的对比可视化
    """
    methods = ['linear', 'cosine', 'sigmoid']
    fig, axes = plt.subplots(len(methods), 10, figsize=(20, len(methods) * 2))
    
    for method_idx, method in enumerate(methods):
        images, _ = advanced_interpolation_visual(generator, start_label, end_label, 
                                                num_step=10, interpolation_method=method, 
                                                save_individual=False)
        
        for i in range(10):
            img = images[i].cpu().squeeze()
            enhanced_img = enhance_image_quality(img, target_size=64)
            
            axes[method_idx, i].imshow(enhanced_img, cmap='gray', interpolation='bicubic')
            axes[method_idx, i].axis('off')
            
            if i == 0:
                axes[method_idx, i].set_ylabel(method.capitalize(), fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Interpolation Method Comparison: {start_label} → {end_label}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存对比图
    import os
    os.makedirs('./visual/interpolation/comparison', exist_ok=True)
    plt.savefig(f'./visual/interpolation/comparison/{start_label}_to_{end_label}_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

# 调用插值函数进行测试
if __name__ == "__main__":
    print("开始生成高质量插值可视化...")

    # 高质量插值
    for i in range(0, 9):
        print(f"生成插值: {i} → {i+1}")
        advanced_interpolation_visual(generator, start_label=i, end_label=i+1, 
                                    num_step=15, interpolation_method='cosine')
    
    # 生成一些特殊的对比
    print("生成插值方法对比...")
    create_comparison_visualization(generator, start_label=1, end_label=6)
    create_comparison_visualization(generator, start_label=0, end_label=9)
    
    print("高质量插值可视化完成！")