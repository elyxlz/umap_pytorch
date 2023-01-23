Parametric UMAP port for pytorch using pytorch lightning for the training loop.

## Install
```bash
pip install umap-pytorch
```

## Usage

```py
from umap_pytorch import PUMAP

pumap = PUMAP(
        encoder=None,           # nn.Module, None for default
        decoder=None,           # nn.Module, True for default, None for encoder only
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,               # How much to weigh reconstruction loss for decoder
        random_state=None,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        num_workers=1,
        num_gpus=1,
        match_nonparametric_umap=False # Train network to match embeddings from non parametric umap
)

data = torch.randn((50000, 512))
pumap.fit(data)
embedding = pumap.transform(data) # (50000, 2)

# if decoder enabled
recon = pumap.inverse_transform(embedding)  # (50000, 512)
```

## Saving and Loading
```py
# Saving
path = 'some/path/hello.pkl'
pumap.save(path)

# Loading
from umap_pytorch import load_pumap
pumap = load_pumap(path)
```
