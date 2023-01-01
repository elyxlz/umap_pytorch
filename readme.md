Parametric UMAP port for pytorch using pytorch lightning for the training loop.

## Install
```bash
pip install umap-pytorch
```

## Usage

```py
from umap_pytorch import PUMAP, conv

encoder = conv(n_components=2)
pumap = PUMAP(
        encoder,              # pytorch encoder module
        decoder=None,         # pytorch decoder module
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        lr=1e-3,
        epochs=30,
        batch_size=64,
        num_workers=1,
        random_state=None
)

data = torch.randn((50000, 512))
pumap.fit(data)
embedding = pumap.transform(data) # (50000, 2)
```
