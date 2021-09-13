import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from architecture.model import Net
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


transform = transforms.Compose([
        transforms.RandomResizedCrop(32, (0.5, 2.0)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

net = Net()
m_state_dict = torch.load('/content/firstCNN/cifar_net.pth')
net.load_state_dict(m_state_dict)

img = Image.open("/content/firstCNN/images/cano.jpg")
img = Image.open("/content/firstCNN/images/caro.jpg")
img = Image.open("/content/firstCNN/images/frogo.jpg")
img = img.resize((32,32))

plt.imshow(img)
plt.show()


img = transform(img)
img = img.unsqueeze(0)
print(img.shape)
print(img)

net.eval()
output = net(img)
print(output)
print(output.shape)






