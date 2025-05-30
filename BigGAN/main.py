import torch
import torch.nn as nn
from mini_biggan import MiniBigGANGenerator, MiniBigGANDiscriminator
from .utils import get_animalfaces_dataloader
from .train import train_miniBigGAN


def main():
    # 检测可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张GPU")
    
    num_epochs = 300
    batch_size = 128 * num_gpus  # 根据GPU数量调整batch size
    image_size = 512
    num_classes = 3
    num_workers = 8 * num_gpus  # 增加数据加载线程数
    img_channels = 3
    z_dim = 128
    base_channels = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用batch_size: {batch_size}, num_workers: {num_workers}")

    # 数据加载器
    dataloader = get_animalfaces_dataloader(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )

    # 生成器
    generator = MiniBigGANGenerator(
        z_dim=z_dim,
        num_classes=num_classes,
        channels=img_channels,
        base_channels=base_channels,
        use_shared_embedding=True,
        hierarchical_z=True
    ).to(device)

    # EMA生成器
    ema_generator = MiniBigGANGenerator(
        z_dim=z_dim,
        num_classes=num_classes,
        channels=img_channels,
        base_channels=base_channels,
        use_shared_embedding=True,
        hierarchical_z=True
    ).to(device)

    # 判别器
    discriminator = MiniBigGANDiscriminator(
        channels=img_channels,
        num_classes=num_classes,
        base_channels=base_channels
    ).to(device)
    
    # 使用DataParallel包装模型进行多GPU并行
    if num_gpus > 1:
        print(f"启用多GPU并行训练，使用 {num_gpus} 张GPU")
        generator = nn.DataParallel(generator)
        ema_generator = nn.DataParallel(ema_generator)
        discriminator = nn.DataParallel(discriminator)
    
    # 训练
    train_miniBigGAN(
        generator=generator,
        discriminator=discriminator,
        ema_generator=ema_generator,
        device=device,
        dataloader=dataloader,
        num_epochs=num_epochs,
        num_classes=num_classes,
        num_gpus=num_gpus,  # 传递GPU数量信息
    )

    

if __name__ == "__main__":
    main()