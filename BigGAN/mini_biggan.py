import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """SA-GAN风格的自注意力模块"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # SA-GAN使用1/8的通道数进行query和key的投影
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # 输出投影层
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # 可学习的注意力权重参数，初始化为0
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 使用谱归一化
        self.apply_spectral_norm()
        
    def apply_spectral_norm(self):
        """应用谱归一化到卷积层"""
        self.query_conv = nn.utils.spectral_norm(self.query_conv)
        self.key_conv = nn.utils.spectral_norm(self.key_conv)
        self.value_conv = nn.utils.spectral_norm(self.value_conv)
        self.out_conv = nn.utils.spectral_norm(self.out_conv)
        
    def forward(self, x):
        """
        SA-GAN自注意力前向传播
        """
        batch_size, channels, height, width = x.size()
        
        # 生成query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # 计算注意力矩阵
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # 应用注意力到value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # 通过输出投影层
        out = self.out_conv(out)
        
        # 残差连接，使用可学习的gamma参数
        out = self.gamma * out + x
        
        return out

class ConditionalBatchNorm2d(nn.Module):
    """条件批标准化 - BigGAN风格"""
    def __init__(self, num_features, num_classes, eps=1e-5):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.eps = eps
        
        # 使用不带仿射变换的批标准化
        self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps)
        
        # 类别条件的gamma和beta嵌入层
        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta = nn.Embedding(num_classes, num_features)
        
        # BigGAN风格的初始化
        nn.init.ones_(self.embed_gamma.weight)
        nn.init.zeros_(self.embed_beta.weight)
        
    def forward(self, x, y):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
            y: 类别标签 [B]
        """
        # 批标准化
        out = self.bn(x)
        
        # 获取类别特定的gamma和beta
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        
        # 应用类别条件的仿射变换
        return gamma * out + beta

class GeneratorBlock(nn.Module):
    """BigGAN生成器残差块 - 改进版"""
    def __init__(self, in_channels, out_channels, num_classes, upsample=True, z_chunk_dim=0):
        super(GeneratorBlock, self).__init__()
        self.upsample = upsample
        self.z_chunk_dim = z_chunk_dim

        # 如果有z_chunk_dim，需要注入z_chunk
        if z_chunk_dim > 0:
            self.z_chunk_linear = nn.Linear(self.z_chunk_dim, out_channels)

        # 主路径的卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 条件批标准化层
        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        
        # 跳跃连接
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        
        # 应用谱归一化
        self.apply_spectral_norm()
        
    def apply_spectral_norm(self):
        """为生成器应用谱归一化"""
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.shortcut = nn.utils.spectral_norm(self.shortcut)
        
    def forward(self, x, y, z_chunk=None):
        # 主路径
        h = self.cbn1(x, y)
        h = F.relu(h)
        
        # 上采样（如果需要）
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            
        h = self.conv1(h)
        h = self.cbn2(h, y)
        h = F.relu(h)
        
        # 如果有z_chunk_dim，在conv2之前注入z_chunk
        if z_chunk is not None and self.z_chunk_dim > 0:
            z_feature = self.z_chunk_linear(z_chunk) # [B, out_channels]
            z_feature = z_feature.unsqueeze(-1).unsqueeze(-1)  # [B, out_channels, 1, 1]
            h = h + z_feature # 广播相加到conv2的输入
            
        h = self.conv2(h)

        # 跳跃连接路径
        x_shortcut = x
        if self.upsample:
            x_shortcut = F.interpolate(x_shortcut, scale_factor=2, mode='nearest')
        x_shortcut = self.shortcut(x_shortcut)
        
        return h + x_shortcut

class MiniBigGANGenerator(nn.Module):
    """改进的MiniBigGAN生成器 - 完整BigGAN架构"""
    def __init__(self, z_dim=128, num_classes=10, channels=3, base_channels=64, 
             use_shared_embedding=True, hierarchical_z=True):
        super(MiniBigGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.use_shared_embedding = use_shared_embedding
        self.hierarchical_z = hierarchical_z
        
        # BigGAN的分层z向量
        if hierarchical_z:
            self.z_chunks = 5
            # 计算每个chunk的实际大小
            base_chunk_size = z_dim // self.z_chunks
            remainder = z_dim % self.z_chunks
            self.z_chunk_sizes = [base_chunk_size + (1 if i < remainder else 0) for i in range(self.z_chunks)]
            # [26, 26, 26, 25, 25] for z_dim=128
        else:
            self.z_chunks = 1
            self.z_chunk_sizes = [z_dim]
        
        # 共享类别嵌入（BigGAN的关键特性）
        if use_shared_embedding:
            self.shared_embedding = nn.Embedding(num_classes, 128)
            
        # 初始线性层 -> 4x4特征图
        input_dim = self.z_chunk_sizes[0] + (128 if use_shared_embedding else 0)
        self.linear = nn.Linear(input_dim, base_channels * 8 * 4 * 4)
        
        # 生成器块序列 - 为每个块分配对应的z_chunk_dim
        if hierarchical_z:
            self.blocks = nn.ModuleList([
                GeneratorBlock(base_channels * 8, base_channels * 8, num_classes, z_chunk_dim=self.z_chunk_sizes[1]),  # 使用z_chunks[1]
                GeneratorBlock(base_channels * 8, base_channels * 4, num_classes, z_chunk_dim=self.z_chunk_sizes[2]),  # 使用z_chunks[2]
                GeneratorBlock(base_channels * 4, base_channels * 2, num_classes, z_chunk_dim=self.z_chunk_sizes[3]), # 使用z_chunks[3]
                GeneratorBlock(base_channels * 2, base_channels, num_classes, z_chunk_dim=self.z_chunk_sizes[4]),     # 使用z_chunks[4]
                GeneratorBlock(base_channels, base_channels // 2, num_classes, z_chunk_dim=0),                       # 最后一层不注入
            ])
        else:
            # 非分层版本
            self.blocks = nn.ModuleList([
                GeneratorBlock(base_channels * 8, base_channels * 8, num_classes),
                GeneratorBlock(base_channels * 8, base_channels * 4, num_classes),
                GeneratorBlock(base_channels * 4, base_channels * 2, num_classes),
                GeneratorBlock(base_channels * 2, base_channels, num_classes),
                GeneratorBlock(base_channels, base_channels // 2, num_classes),
            ])
        
        # 自注意力层 - 在64x64和32x32都添加
        self.attention_64 = SelfAttention(base_channels)
        self.attention_32 = SelfAttention(base_channels * 2)
        
        # 最终输出层
        self.final_bn = nn.BatchNorm2d(base_channels // 2)
        self.final_conv = nn.Conv2d(base_channels // 2, channels, 3, padding=1)
        
        # 对最终卷积层应用谱归一化
        self.final_conv = nn.utils.spectral_norm(self.final_conv)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """BigGAN风格的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)
                
    def forward(self, z, y):
        # 预先处理z_chunks
        if self.hierarchical_z:
            z_chunks = torch.split(z, self.z_chunk_sizes, dim=1)
        else:
            z_chunks = [z]  # 非分层模式下也创建列表以保持一致性
        
        if self.use_shared_embedding:
            y_embed = self.shared_embedding(y)
            h_input = torch.cat([z_chunks[0], y_embed], dim=1)
            # print(f"y_embed.shape: {y_embed.shape}")  # 调试输出
            # print(f"z_chunks[0].shape: {z_chunks[0].shape}")
            # print(f"h_input.shape: {h_input.shape}")  # 调试输出
        else:
            h_input = z_chunks[0]
    
        for i in range(len(z_chunks)):
            pass
            # print(f"z_chunks[{i}].shape: {z_chunks[i].shape}")  # 调试输出


        # 将噪声向量重塑为4x4特征图
        h = self.linear(h_input)
        h = h.view(h.size(0), self.base_channels * 8, 4, 4)
        
        # 通过生成器块逐步上采样
        for i, block in enumerate(self.blocks):
            # 准备z_chunk用于注入
            z_chunk = None
            if self.hierarchical_z and (i + 1) < len(z_chunks):
                z_chunk = z_chunks[i + 1]
            
            # 传递z_chunk到GeneratorBlock
            h = block(h, y, z_chunk)
            
            # 在特定分辨率添加自注意力
            if h.shape[-1] == 32:  # 32x32分辨率
                h = self.attention_32(h)
            elif h.shape[-1] == 64:  # 64x64分辨率
                h = self.attention_64(h)
    
        # 最终输出
        h = self.final_bn(h)
        h = F.relu(h)
        h = self.final_conv(h)
        h = torch.tanh(h)  # 输出范围[-1, 1]
        
        return h

class DiscriminatorBlock(nn.Module):
    """改进的判别器残差块"""
    def __init__(self, in_channels, out_channels, downsample=True, first_block=False):
        super(DiscriminatorBlock, self).__init__()
        self.downsample = downsample
        self.first_block = first_block
        
        # 预激活残差块
        if not first_block:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 跳跃连接
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = None
            
        if downsample:
            self.avg_pool = nn.AvgPool2d(2)
            
        # 应用谱归一化
        self.apply_spectral_norm()
            
    def apply_spectral_norm(self):
        """应用谱归一化"""
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        if self.shortcut:
            self.shortcut = nn.utils.spectral_norm(self.shortcut)
            
    def forward(self, x):
        h = x
        
        # 预激活
        if not self.first_block:
            h = self.bn1(h)
            h = self.relu1(h)
            
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = self.avg_pool(h)
            
        # 跳跃连接
        if self.shortcut:
            x_shortcut = self.shortcut(x)
            if self.downsample:
                x_shortcut = self.avg_pool(x_shortcut)
        else:
            x_shortcut = x
            if self.downsample:
                x_shortcut = self.avg_pool(x_shortcut)
                
        return h + x_shortcut

class MiniBigGANDiscriminator(nn.Module):
    """改进的MiniBigGAN判别器 - 投影判别器"""
    def __init__(self, num_classes=10, channels=3, base_channels=64):
        super(MiniBigGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # 判别器块序列
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(channels, base_channels, first_block=True),           # 128->64
            DiscriminatorBlock(base_channels, base_channels * 2),                    # 64->32
            DiscriminatorBlock(base_channels * 2, base_channels * 4),                # 32->16
            DiscriminatorBlock(base_channels * 4, base_channels * 8),                # 16->8
            DiscriminatorBlock(base_channels * 8, base_channels * 16),               # 8->4
            DiscriminatorBlock(base_channels * 16, base_channels * 16, downsample=False), # 4x4
        ])
        
        # 自注意力层 - 在多个分辨率添加
        self.attention_32 = SelfAttention(base_channels * 2)
        self.attention_16 = SelfAttention(base_channels * 4)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 输出层 - 真假判别
        self.linear = nn.Linear(base_channels * 16, 1)
        
        # 投影判别器 - BigGAN的关键特性
        self.embedding = nn.Embedding(num_classes, base_channels * 16)
        
        # 应用谱归一化
        self.linear = nn.utils.spectral_norm(self.linear)
        self.embedding = nn.utils.spectral_norm(self.embedding)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)
        
    def forward(self, x, y=None):
        h = x
        
        # 通过判别器块
        for i, block in enumerate(self.blocks):
            h = block(h)
            
            # 在特定分辨率添加自注意力
            if h.shape[-1] == 32:  # 32x32分辨率
                h = self.attention_32(h)
            elif h.shape[-1] == 16:  # 16x16分辨率
                h = self.attention_16(h)
        
        # 全局池化
        h = self.global_pool(h)
        h = h.view(h.size(0), -1)
        
        # 真假判别
        output = self.linear(h)
        
        # 投影判别 - 如果提供标签
        if y is not None:
            y_embed = self.embedding(y)
            projection = torch.sum(h * y_embed, dim=1, keepdim=True)
            output = output + projection
            
        return output

if __name__ == "__main__":
    # 测试参数
    batch_size = 4
    z_dim = 128
    num_classes = 10
    img_size = 128
    channels = 3

    # 随机输入
    z = torch.randn(batch_size, z_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # 生成器
    G = MiniBigGANGenerator(z_dim=z_dim, num_classes=num_classes, channels=channels)
    fake_imgs = G(z, y)
    print("生成器输出形状:", fake_imgs.shape)  # 期望: [batch_size, channels, img_size, img_size]

    # 判别器
    D = MiniBigGANDiscriminator(num_classes=num_classes, channels=channels)
    out = D(fake_imgs, y)
    print("判别器输出形状:", out.shape)  # 期望: [batch_size, 1]