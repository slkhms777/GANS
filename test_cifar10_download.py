import os

def check_cifar10_download(data_path='./data'):
    cifar10_dir = os.path.join(data_path, 'cifar-10-batches-py')
    required_files = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
        'test_batch',
        'batches.meta'
    ]
    if not os.path.isdir(cifar10_dir):
        print(f"目录不存在: {cifar10_dir}")
        return False

    missing = []
    for fname in required_files:
        fpath = os.path.join(cifar10_dir, fname)
        if not os.path.isfile(fpath):
            missing.append(fname)
    if missing:
        print("缺少以下CIFAR-10文件:")
        for fname in missing:
            print("  -", fname)
        return False
    else:
        print("CIFAR-10数据集已完整下载！")
        return True

if __name__ == '__main__':
    check_cifar10_download('./data')