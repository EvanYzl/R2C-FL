# Dataset layout

The datasets are not redistributed in this repository. Download them from the
official sources and place them under this directory before running
`python -m r2c_baselines.prepare`.

| Dataset | Directory | Official source | Train/test split |
|---|---|---|---|
| Fashion-MNIST | `fashion_mnist/` | Zalando Research Fashion-MNIST | 60,000 / 10,000 |
| CIFAR-10 | `cifar10/` | Toronto CIFAR | 50,000 / 10,000 |
| SVHN | `svhn/` | Stanford UFLDL SVHN | 73,257 / 26,032 |
| CIFAR-100 | `cifar100/` | Toronto CIFAR | 50,000 / 10,000 |

The extra SVHN split is not used. The loader uses `download=False` so that a
formal run cannot silently replace the audited local dataset. The preparation
step validates the local content and binds its checksum into the run lineage.
