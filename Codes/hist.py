import torch
from utils.datasets import all_datasets
import matplotlib.pyplot as plt 
import numpy as np

from utils.resnet_duq import ResNet_DUQ


ds3 = all_datasets["CIFAR10"]()
input_size, num_classes, _ , test_dataset_CIFAR = ds3

model_CIFAR = ResNet_DUQ(input_size, num_classes, 512 ,512 ,0.1 ,0.999)


ds4 = all_datasets["SVHN"]()
_,_,_, test_dataset_SVHN = ds4

model_CIFAR.load_state_dict(torch.load("/content/DUQ_CIFAR_75.pt"))
model_CIFAR.eval()

kernel_distace_CIFAR = np.zeros((len(test_dataset_CIFAR),1))

for i in range(len(test_dataset_CIFAR)):
  kernel_distace_CIFAR[i] = model_CIFAR(test_dataset_CIFAR[i][0].reshape(1,3,32,32))[1].max(1)[0].item()


kernel_distace_SVHN = np.zeros((len(test_dataset_SVHN),1))

for i in range(len(test_dataset_SVHN)):
  kernel_distace_SVHN[i] = model_CIFAR(test_dataset_SVHN[i][0].reshape(1,3,32,32))[1].max(1)[0].item()
  

n, bins, patches = plt.hist(x=kernel_distace_SVHN, bins='auto', color='#0504aa'
                           )

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Certainity')

maxfreq = n.max()
print(maxfreq)

plt.ylim(0,maxfreq)
