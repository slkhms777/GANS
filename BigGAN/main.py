import torch
from mini_biggan import MiniBigGANGenerator, MiniBigGANDiscriminator
from .utils import get_animalfaces_dataloader
from .train import train_miniBigGAN


def main():
    num_epochs = 300
    batch_size = 128
    image_size = 512
    num_classes = 3
    num_workers = 8
    img_channels = 3
    z_dim = 128
    base_channels = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # 训练
    train_miniBigGAN(
        generator=generator,
        discriminator=discriminator,
        ema_generator=ema_generator,
        device=device,
        dataloader=dataloader,
        num_epochs=num_epochs,
        num_classes=num_classes,
    )

    

if __name__ == "__main__":
    main()