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


""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        model: nn.Module,
        n_components=2,
        min_dist=0.1,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.n_components = n_components
        self.min_dist = min_dist

    def configure_optimizers(self): 
        return torch.optim.AdamW(self.parameters(), lr=self.lr,)

    def training_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.model(edges_to_exp), self.model(edges_from_exp)
        loss = umap_loss(embedding_to, embedding_from, edges_to_exp.shape[0], self.min_dist)
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
        model,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        lr = 1e-5,
        epochs = 50,
        batch_size=1024,
        num_workers=1,
        random_state = None,
    ):
        self.model = model
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.lr = lr
        self.epochs = 50
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        
    def fit(self, X):
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
        graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
        trainer.fit(
            model=Model(self.lr, self.model, n_components=self.n_components, min_dist=self.min_dist),
            datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
            )


if __name__== "__main__":
    from torchvision.datasets import MNIST
    import torchvision
    from torchvision.transforms import transforms
    
    a = Model(1e-5, conv(2), n_components=2, min_dist=0.1)
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_tensor = torch.stack([example[0] for example in train_dataset])[:, 0][:, None, ...]
    X = train_tensor

    PUMAP = PUMAP(conv(2))
    PUMAP.fit(X)