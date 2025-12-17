# tigerencode

Utilities for encoding images and text with multiple backends.

## Installation

Install the latest version directly from GitHub with pip. All backends and their dependencies are included by default:

```bash
pip install https://github.com/htylab/tigerencode/archive/main.zip
```

## Usage

```python
import tigerencode

image_model = tigerencode.model_img(model="timm@resnet50")
features = image_model.encode_image("/path/to/image.jpg")

features_batch, paths = image_model.encode_image_batch([
    "/path/to/image1.jpg",
    "/path/to/image2.jpg",
])

text_model = tigerencode.model_text()
text_features = text_model.encode_text("hello, world!")

text_features_batch, texts = text_model.encode_text_batch([
    "hello, world!",
    "I am good",
])

# Use a Hugging Face model that requires authentication
private_text_model = tigerencode.model_text(
    model="hf@your-org/your-private-model",
    token="hf_xxx-your-token",
)

# Optionally attach an adaptor implemented as a torch.nn.Module
from torch import nn

projection = nn.Linear(768, 768, bias=False)
image_model.set_adaptor(projection)

# Or supply an ONNX adaptor path (converted via `onnx2torch`)
image_model_with_onnx = tigerencode.model_img(
    model="timm@resnet50",
    adaptor="/path/to/adaptor.onnx",
)

# Use the built-in ProjectionAdaptor for feature normalization
from tigerencode.adaptor import ProjectionAdaptor

projection_adaptor = ProjectionAdaptor(in_dim=1024, out_dim=768)
image_model.set_adaptor(projection_adaptor)
```

When an ONNX file path is provided, TigerEncode will lazily convert it to a PyTorch
module using [`onnx2torch`](https://github.com/onnx/onnx2torch), which is installed
alongside the package.

### Clustering and deduplication utilities

> Clustering relies on `python-igraph` and `leidenalg`, which are installed as part of
> the package dependencies.

```python
import tigerencode
import numpy as np

# Leiden clustering over embeddings
embeddings = np.random.rand(1000, 128).astype(np.float32)
cluster_ids = tigerencode.embed_clustering_leiden(
    embeddings,
    topk=150,
    resolutions=[0.5, 1.0, 1.5],
    verbose=False,
)

# Strict deduplication with iterative mutual-k merging
cluster_id, merged_embeddings, cluster_size, rep_index, info = tigerencode.knn_merge(
    embeddings,
    mutual_k=3,
    merge_knn_topk=50,
    merge_top_p=0.01,
    merge_sim_min=0.92,
    max_cluster_size=100,
    verbose=True,
)

# `strict_dedup` remains available as a compatibility alias.
```
