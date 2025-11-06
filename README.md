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
```
