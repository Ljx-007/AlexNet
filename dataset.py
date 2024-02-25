import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import Lambda

# 对训练集和测试集进行数据增强
train_tran = transforms.Compose([
    transforms.Resize(256),  # 先把32*32的数据集变成256*256大小
    transforms.RandomHorizontalFlip(),  # 将图片水平反转，垂直反转用RandomVerticalFlip
    transforms.RandomCrop(224),  # 随即裁剪成224*224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
test_tran = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    # TenCrop太耗算力
    # transforms.TenCrop(224),  # 在图的四个角以及中心（包括水平反转）裁剪出224*224大小，一共十个部位，所以为TenCrop
    # TenCrop是返回了10张图片，要将这10张四维图片(batchsize,c,h,w)合成一个tensor，这个tensor就是五维的(batchsize,ncrops,c,h,w)
    # 使用transforms.Lambda来自定义将10个四维tensor变成五维tensor
    # transforms.Lambda是pytorch给出的自定义transforms操作
    # crop 为10张剪裁图的一张图，crops是Tencrop返回的10张图，使用for循环将每张图都转为tensor型，然后用torch.stack把这10个tensor都拼起来，变成5维的
    # Lambda(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops]))
])
train_dataset = torchvision.datasets.CIFAR100("CIFAR100", True, transform=train_tran, download=True)
test_dataset = torchvision.datasets.CIFAR100("CIFAR100", False, transform=test_tran, download=True)
train_load = DataLoader(train_dataset, 128)
test_load = DataLoader(test_dataset, 128)
if __name__ == '__main__':
    # write = SummaryWriter("dataset")
    # step = 0
    print(len(train_dataset))
    # 用tensorboard看看
    for data in test_load:
        img, target = data
        bs, ncops, c, h, w = img.size()
        img = img.mean(1)
        # write.add_images("train_data", img, step, dataformats="NCHW")
        print(img.shape)
        break
    # write.close()
