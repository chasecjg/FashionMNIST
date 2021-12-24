from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [transforms.Resize((28, 28)),
     transforms.Grayscale(1),
     # transforms.ToTensor()
     # transforms.Normalize((0.5,), (0.5,))
     ])

im = Image.open('./test_images/5.png')

im = transform(im)  # [C, H, W]
# im.show()
im.save("./test_images/005.jpg")