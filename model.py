import torch
from torch import nn


class Alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 提取特征
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),  # 224*224
            # 卷积完Relu然后maxpool，size变小后卷积时用same padding，这样size不会变化
            # 对z值做归一化，不然模型训练不动
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 27*27
            nn.Conv2d(96, 256, 5, padding=2),  # 27*27
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 13*13
            nn.Conv2d(256, 384, 3, 1, 1),  # 13*13
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1),  # 13*13
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, 1),  # 13*13
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),  # 6*6
            nn.Flatten())  # 9216向量
        # self.init_weight()
        # 全连接层
        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 100)
        )

    # def init_weight(self):
    #     for m in self.modules():  # self.modules（）可以遍历模型中的所有模块
    #         # isinstance是判断对象 m 是否是Conv2d类型，如果是，返回True，否则False
    #         if isinstance(m, nn.Conv2d):  # uniform是均匀分布初始化，normal是正态分布初始化，一般用normal
    #             nn.init.normal_(m.weight, mean=0, std=0.01)
    #         elif isinstance(m, nn.Linear):  # 线性层一般用Xavier初始化，卷积层一般不用，卷积层中的技术已经提供了足够的稳定性
    #             nn.init.xavier_normal_(m.weight)
    #             # 全连接层的bias都是1
    #             nn.init.constant_(m.bias, 1)  # constant函数用来初始化bias
    #         # AlexNet还指定了第2，4，5层的卷积的bias为1，其余为0
    #     nn.init.constant_(self.feature[3].bias, 1)
    #     nn.init.constant_(self.feature[8].bias, 1)
    #     nn.init.constant_(self.feature[10].bias, 1)

    def forward(self, x):
        x = self.feature(x)
        x = self.dense(x)
        return x

    # 是否加载预训练数据,默认为不加载，无路径
    def pre_alex(self, pretrain=False, root=None):
        model = Alexnet()
        if pretrain:
            model = model.load_state_dict(torch.load(root))
        return model


if __name__ == '__main__':
    model = Alexnet()
    x = torch.ones((5, 3, 224, 224))
    y = model(x)
    target = torch.tensor([6, 25, 14, 15, 85])
    pred = (y.argmax(1) == target)
    y = pred.sum()
    z = y.item()
    print(pred)
    print(y)
    print(z)
