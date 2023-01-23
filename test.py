import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from umap_pytorch import PUMAP, load_pumap
import seaborn as sns
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_tensor = torch.stack([example[0] for example in train_dataset])[:, 0][:, None, ...]
labels = [str(example[1]) for example in train_dataset]
X = train_tensor

pumap = PUMAP(epochs=5, min_dist=1, n_neighbors=50, num_workers=8, decoder=True, beta = 0.01, match_nonparametric_umap=True)
pumap.fit(X)
pumap.save('yo.pkl')
pumap = load_pumap('yo.pkl')
embedding = pumap.transform(X)
print(embedding.shape, embedding)
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=0.4)
plt.savefig('test4.png')


def regenerate_and_plot(i=6):
    some_points = embedding[np.random.choice(embedding.shape[0], 6)]
    regenerated = pumap.inverse_transform(torch.Tensor(some_points))

    for i in range(6):
        img = regenerated[i,0]
        img = Image.fromarray(np.uint8(img))
        img.save("image_{}.png".format(i))
        
regenerate_and_plot()
    