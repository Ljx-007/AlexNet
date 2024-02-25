import time

import torch

from dataset import *
from model import *

# 定义超参数
num_iteration = 20
learning_rate = 0.05
moment = 0.9
weight_decay = 0.0005
# 创建网络实例
alex = Alexnet()
# 设置训练GPU
device = torch.device("cuda")
alex.to(device)
# 定义Loss
Loss = nn.CrossEntropyLoss()
Loss.to(device)
# AlexNet采用SGD梯度下降
optim = torch.optim.SGD(alex.parameters(), lr=learning_rate, momentum=moment, weight_decay=weight_decay)
# 记录时间
start_time = time.time()
write = SummaryWriter("train")
for epoch in range(num_iteration):
    alex.train()
    epoch_time = time.time()
    total_train_loss = 0.  # 每轮的损失
    total_train_accuracy = 0.  # 每轮的准确率
    for data in train_load:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = alex(imgs)
        accuracy = (output.argmax(1) == targets).sum().item()
        loss = Loss(output, targets)
        # 将每个批次的损失都加入total_loss，每个批次的准确预测数量都加入total_accuracy
        total_train_loss += loss
        total_train_accuracy += accuracy
        optim.zero_grad()
        loss.backward()
        optim.step()
    # 每轮都打印Loss和accuracy，计算一轮下来所有批次的平均损失和平均准确率
    train_avg_loss = total_train_loss / len(train_load)  # 因为loss计算的是每个batch的loss，所以计算平均就要除以batch的数量
    train_avg_accuracy = total_train_accuracy / len(train_dataset)  # acc是每个样本的acc预测正确个数的总和，所以要除以总样本的个数
    print(" Train:  epoch:{}  loss:{}  accuracy:{:.2f}%".format(epoch + 1, train_avg_loss, train_avg_accuracy * 100))
    alex.eval()
    total_test_loss = 0.
    total_test_accuracy = 0.
    for data in test_load:
        imgs, targets = data
        # 因为在加载测试集时使用了TenCrop，把四维的imgs变成了五维的(batchsize,ncrops,c,h,w)
        # 但神经网络只接受四维tensor，所以使用它的时候要手动把他变成四维tensor
        # batch_size, ncrops, c, h, w = imgs.size()  # 把imgs的各个维度都拿出来重新排列
        # imgs = imgs.view(-1, c, h, w)  # 把batch_size和ncrops合并起来，此时batchsize变为了原来的ncrop倍，就变成了四维的tensor了
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            # imgs的batchsize变为了原来的ncrop倍，输入网络后得到的特征向量要取均值
            output = alex(imgs)
            # 取均值,把原来的ncrop变回来，剩下-1那部分就是神经网络输出的特征向量
            # output = output.view(batch_size, ncrops, -1).mean(1)  # mean函数对第1维取均值，消掉了ncops的维度，也就是对10个特征向量取了均值
            test_accuracy = (output.argmax(1) == targets).sum().item()
            loss = Loss(output, targets)
            total_test_loss += loss
            total_test_accuracy += test_accuracy
    test_avg_loss = total_test_loss / len(test_load)
    test_avg_accuracy = total_test_accuracy / len(test_dataset)
    print("Test:   epoch:{}  loss:{}  accuracy:{:.2f}%".format(epoch + 1, test_avg_loss, test_avg_accuracy*100))
    current_time = time.time()
    print("Every Epoch Using Time:{:.2f}s".format(current_time - epoch_time))
    print("Totol Time:{:.2f}s".format(current_time - start_time))
torch.save(alex.state_dict(), "../pretrain_param.pth")
write.close()
