import time

import torch
import torchvision
import torchvision.transforms as transforms


from model import LeNet, VGG16
from models.resnet import ResNet


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
         ]
    )

    classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
    val_set = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10,
                                             shuffle=False, num_workers=4)

    # net = LeNet()
    # net = VGG16()
    net = ResNet()
    # net.load_state_dict(torch.load('./weights/lenet/Lenet_45.pth'))
    # net.load_state_dict(torch.load('./weights/vgg/VGG16_45.pth'))
    net.load_state_dict(torch.load('./weights/resnet/resnet_45.pth'))
    net.cuda()
    net.eval()
    start_time = time.time()
    for step, data in enumerate(val_loader):
        input, label = data
        label_numpy = label.cpu().numpy()
        input = input.cuda()
        label = label.cuda()
        with torch.no_grad():
            output = net(input)
            predict = torch.max(output, dim=1)[1]
            predict_numpy = torch.max(output, dim=1)[1].cpu().data.numpy()
            for i in range(10):
                print('predict: %s   label: %s' % (classes[int(predict_numpy[i])], classes[label_numpy[i]]))
            accuracy = torch.eq(predict, label).sum().item() / label.size(0)
            print('iter: %d------------------------------accuracy: %.3f' %(step, accuracy))
    end_time = time.time()
    print("总共测试时间为{}".format(end_time-start_time))


if __name__ == '__main__':
    main()
