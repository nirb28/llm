---
tags:
- t5
- adapter-transformers
datasets:
- glue
---

# Adapter `WillHeld/flan-t5-base-tada-ot` for google/flan-t5-base

An [adapter](https://adapterhub.ml) for the `google/flan-t5-base` model that was trained on the [glue](https://huggingface.co/datasets/glue/) dataset.

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

model = AutoAdapterModel.from_pretrained("google/flan-t5-base")
adapter_name = model.load_adapter("WillHeld/flan-t5-base-tada-ot", source="hf", set_active=True)
```

## Architecture & Training

<!-- Add some description here -->

## Evaluation results

<!-- Add some description here -->

## Citation

<!-- Add some description here -->