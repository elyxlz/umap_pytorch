import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from umap_pytorch import PUMAP, conv
import seaborn as sns
import torch

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_tensor = torch.stack([example[0] for example in train_dataset])[:, 0][:, None, ...]
labels = [str(example[1]) for example in train_dataset]
X = train_tensor

pumap = PUMAP(conv(2), lr=1e-3, epochs=4, num_workers=8)
pumap.fit(X)
embedding = pumap.transform(X)
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=0.4)
plt.savefig('test3.png')
