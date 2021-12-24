import time

import numpy as np
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import LeNet, VGG16
import torch.optim as optim
import torchvision.transforms as transforms

from models.resnet import ResNet


def main():
    # 图像预处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
         ]
    )

    # 60000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=4)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    # classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # # get some random training images
    # dataiter = iter(val_loader)
    # images, labels = dataiter.next()
    # print(labels)
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(1)))


    # net = LeNet()
    # net = VGG16()
    net = ResNet()
    net = net.cuda()
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    start_time = time.time()
    for epoch in range(50):  # loop over the dataset multiple times
        net.train()
        # writer = SummaryWriter("./logs/lenet")
        # writer = SummaryWriter("./logs/vgg16")
        writer = SummaryWriter("./logs/resnet")
        running_loss = 0.0
        # total_train_step = 0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(inputs.shape)
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            net.eval()
            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    val_image = val_image.cuda()
                    val_label = val_label.cuda()
                    outputs = net(val_image)  # [batch, 5000]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    # 绘图
                    # writer.add_scalar("loss_lenet", running_loss/500, epoch)
                    # writer.add_scalar("accuracy_lenet", accuracy, epoch)
                    # writer.add_scalar("loss_vgg", running_loss / 500, epoch)
                    # writer.add_scalar("accuracy_vgg", accuracy, epoch)
                    writer.add_scalar("loss_resnet", running_loss / 500, epoch)
                    writer.add_scalar("accuracy_resnet", accuracy, epoch)
                    running_loss = 0.0
        if epoch % 5 == 0:
            # save_path = './weights/lenet/Lenet_{}.pth'.format(epoch)
            # save_path = './weights/vgg/VGG16_{}.pth'.format(epoch)
            save_path = './weights/resnet/resnet_{}.pth'.format(epoch)
            torch.save(net.state_dict(), save_path)
    end_time = time.time()
    print("总共耗时：{}".format(end_time-start_time))
    print('Finished Training')

if __name__ == '__main__':
    main()
