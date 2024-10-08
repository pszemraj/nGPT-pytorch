## nGPT (normalized GPT) - Pytorch

Quick implementation of <a href="https://arxiv.org/abs/2410.01131">nGPT</a>, learning entirely on the hypersphere, from NvidiaAI. The question is whether there is any loss of expressivity they swept under the rug, but I'll take it with good faith.

## Install

```bash
$ pip install nGPT-pytorch
```

## Usage

```python
import torch
from nGPT_pytorch import nGPT

model = nGPT(
    num_tokens = 256,
    dim = 512,
    depth = 4,
    attn_norm_qk = True
)

x = torch.randint(0, 256, (2, 2048))

loss = model(x, return_loss = True)
loss.backward()

logits = model(x) # (2, 2048, 256)
```

## Citations

```bibtex
@inproceedings{Loshchilov2024nGPTNT,
    title   = {nGPT: Normalized Transformer with Representation Learning on the Hypersphere},
    author  = {Ilya Loshchilov and Cheng-Ping Hsieh and Simeng Sun and Boris Ginsburg},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273026160}
}
```

```bibtex
@article{Luo2017CosineNU,
    title     = {Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks},
    author    = {Chunjie Luo and Jianfeng Zhan and Lei Wang and Qiang Yang},
    journal   = {ArXiv},
    year      = {2017},
    volume    = {abs/1702.05870},
    url       = {https://api.semanticscholar.org/CorpusID:1505432}
}
```
