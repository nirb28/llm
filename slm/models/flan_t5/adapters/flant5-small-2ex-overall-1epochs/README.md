---
tags:
- adapterhub:self-explanations
- t5
- adapter-transformers
datasets:
- self-explanations
---

# Adapter `nbogdan/flant5-small-2ex-overall-1epochs` for google/flan-t5-small

An [adapter](https://adapterhub.ml) for the `google/flan-t5-small` model that was trained on the [self-explanations](https://adapterhub.ml/explore/self-explanations/) dataset.

This adapter was created for usage with the **[adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers)** library.

## Usage

First, install `adapter-transformers`:

```
pip install -U adapter-transformers
```
_Note: adapter-transformers is a fork of transformers that acts as a drop-in replacement with adapter support. [More](https://docs.adapterhub.ml/installation.html)_

Now, the adapter can be loaded and activated like this:

```python
from transformers import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("google/flan-t5-small")
adapter_name = model.load_adapter("nbogdan/flant5-small-2ex-overall-1epochs", source="hf", set_active=True)
```

## Architecture & Training

<!-- Add some description here -->

## Evaluation results

<!-- Add some description here -->

## Citation

<!-- Add some description here -->