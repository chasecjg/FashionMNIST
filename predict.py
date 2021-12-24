# import cv2
import cv2
import mmcv
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet, VGG16
from models.resnet import ResNet


def main():
    transform = transforms.Compose(
        [
         transforms.Resize((28, 28)),
         transforms.Grayscale(1),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
         ])

    classes = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')


    net = LeNet()
    # net = VGG16()
    # net = ResNet()
    net.eval()

    net.load_state_dict(torch.load('./weights/lenet/Lenet_45.pth'))
    # net.load_state_dict(torch.load('./weights/vgg/VGG16_45.pth'))
    # net.load_state_dict(torch.load('./weights/resnet/resnet_45.pth'))
    path = "./test_images/2.png"
    im = Image.open(path)
    # im = cv2.imread(path)

    im = transform(im)  # [C, H, W]
    # Image._show(im)
    # print(im.shape)
    # print(im.mode)
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
    # print(im.shaoe)
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
