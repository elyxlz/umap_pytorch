import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.functional import mse_loss
#from umap_pytorch.modules import umap_loss, get_umap_graph

from model import conv
from data import UMAPDataset
from modules import umap_loss, get_umap_graph
from umap.umap_ import find_ab_params


""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        min_dist=0.1,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
        loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0])
        self.log("train_loss", loss)
        return loss


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,        
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

class PUMAP():
    def __init__(
        self,
        encoder,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        lr=1e-3,
        epochs=30,
        batch_size=64,
        num_workers=1,
        random_state=None,
    ):
        self.encoder = encoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        
    def fit(self, X):
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
        self.model = Model(self.lr, self.encoder, min_dist=self.min_dist)
        graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
        trainer.fit(
            model=self.model,
            datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
            )
    @torch.no_grad()
    def transform(self, X):
        self.embedding_ = self.model.encoder(X).detach().cpu().numpy()
        return self.embedding_
        


if __name__== "__main__":
    import torchvision
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_tensor = torch.stack([example[0] for example in train_dataset])[:, 0][:, None, ...]
    labels = [str(example[1]) for example in train_dataset]
    X = train_tensor

    PUMAP = PUMAP(conv(2), epochs=4, num_workers=8)
    PUMAP.fit(X)
    import seaborn as sns
    embedding = PUMAP.transform(X)
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=0.4)
    plt.savefig('test2.png')