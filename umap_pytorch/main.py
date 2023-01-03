import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

from umap_pytorch.data import UMAPDataset
from umap_pytorch.modules import get_umap_graph, umap_loss
from umap_pytorch.model import default_encoder, default_decoder

from umap.umap_ import find_ab_params
import dill


""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta = 1.0,
        min_dist=0.1,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta # weight for reconstruction loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        (edges_to_exp, edges_from_exp) = batch
        embedding_to, embedding_from = self.encoder(edges_to_exp), self.encoder(edges_from_exp)
        encoder_loss = umap_loss(embedding_to, embedding_from, self._a, self._b, edges_to_exp.shape[0], negative_sample_rate=5)
        self.log("umap_loss", encoder_loss)
        
        if self.decoder is not None:
            recon = self.decoder(embedding_to)
            recon_loss = mse_loss(recon, edges_to_exp)
            self.log("recon_loss", recon_loss)
            return encoder_loss + self.beta * recon_loss
        else:
            return encoder_loss


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
        encoder=None,
        decoder=None,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,
        random_state=None,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        num_workers=1,
        num_gpus=1,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus

    def fit(self, X):
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs)
        encoder = default_encoder(X.shape[1:], self.n_components) if self.encoder is None else self.encoder
        
        if self.decoder is None or isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder == True:
            decoder = default_decoder(X.shape[1:], self.n_components)
            
        self.model = Model(self.lr, encoder, decoder, beta=self.beta, min_dist=self.min_dist)
        graph = get_umap_graph(X, n_neighbors=self.n_neighbors, metric=self.metric, random_state=self.random_state)
        trainer.fit(
            model=self.model,
            datamodule=Datamodule(UMAPDataset(X, graph), self.batch_size, self.num_workers)
            )
        
    @torch.no_grad()
    def transform(self, X):
        print(f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})")
        return self.model.encoder(X).detach().cpu().numpy()
    
    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()
    
    def save(self, path):
        with open(path, 'wb') as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")
        
def load_pumap(path): 
    print("Loading PUMAP object from pickled file.")
    with open(path, 'rb') as inp: return dill.load(inp)

if __name__== "__main__":
    pass