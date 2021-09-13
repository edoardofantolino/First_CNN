import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from architecture.model import Net
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

lista = ['airplane', 'car', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck']

net = Net()
m_state_dict = torch.load('/content/firstCNN/cifar_net.pth')
net.load_state_dict(m_state_dict)

img = Image.open("/content/firstCNN/images/cano.jpg")
img = Image.open("/content/firstCNN/images/caro.jpg")
img = Image.open("/content/firstCNN/images/frogo.jpg")
img = img.resize((32,32))

plt.imshow(img)
plt.title("The image")
plt.show()

img = transform(img)
img = img.unsqueeze(0)
# print(img.shape)
# print(img)
print()

net.eval()
output = net(img)
# print(output)
# print(output.shape)

x = np.array([0,1,2,3,4,5,6,7,8,9])
plt.bar(x, output[0].detach())
plt.title("Output of the model")
plt.xticks(range(len(lista)), lista)
plt.grid()
plt.show()

print()

softmax = nn.Softmax(dim=1)
output = softmax(output)
plt.bar(x, output[0].detach())
plt.title("Softmax(output)")
plt.xticks(range(len(lista)), lista)
plt.grid()
plt.show()

print()




