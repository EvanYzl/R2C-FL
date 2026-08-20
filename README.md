# R2C-FL

> **Review-only artifact.** This repository is provided solely for peer review
> and reproducibility assessment of the accompanying manuscript. It is not a
> production release. Third-party snapshots remain governed by their original
> licenses, as detailed in `THIRD_PARTY.md`.

Official reproduction code for **“R2C-FL: Auditable History-Balanced Client
Selection and Dual-Timescale Deployment Control under Compound Drift.”**

This repository contains the complete executable harness used for every
reported R2C-FL run and for all **77 reproduced baseline runs**. The seven
baseline adapters are FedAvg, FedAU, F3AST, FedAWE, Power-of-Choice, Oort, and
TiFL. No value in the paper tables is copied from a source paper.

## What is included

- `r2c_baselines/`: the common training harness, all seven baseline adapters,
  R2C-FL, queue builders, aggregation, metrics, and audit/finalization code;
- `tests/`: unit, queue, metric, and run-audit tests;
- `frozen_assets/`: frozen client partitions and dynamic traces used by the
  matched protocol;
- `manifests/`: the immutable baseline, four-dataset, and ablation manifests;
- `upstream/`: read-only snapshots used to check selected adapter semantics.
  The matched runs execute the common adapters in `r2c_baselines/`, not the
  heterogeneous upstream training stacks.

Raw datasets, checkpoints, and the approximately 2.2 GB local `runs/`
directory are intentionally not redistributed. Dataset locations and sources
are documented in `data/raw/README.md`.

## Environment

The released code was validated with Python 3.11, PyTorch 2.12, CUDA 13.0,
and the package versions in `requirements.txt`. Install the PyTorch build that
matches the local CUDA driver, then install the remaining requirements.

```bash
python -m pip install -r requirements.txt
```

## Reproducing the baseline matrix

Run commands from the repository root after placing the four datasets under
`data/raw/`.

```bash
python -m r2c_baselines.prepare
python -m r2c_baselines.queue build
python -m r2c_baselines.queue worker --stop-after-stage formal
python -m r2c_baselines.aggregate
```

The generated formal block contains 56 S0/S4 runs across four datasets and
seven methods, plus 21 CIFAR-10 S1/S2/S3 runs: 77 baseline runs in total.

## Reproducing the reported R2C-FL runs

The matched four-dataset S0/S4 matrix is built and executed with:

```bash
python -m r2c_baselines.r2c_v11_four_dataset_matched_queue build
python -m r2c_baselines.r2c_v11_four_dataset_matched_queue worker
python -m r2c_baselines.r2c_v11_four_dataset_matched_finalize
```

The additional CIFAR-10 drift-factor and component-ablation runs are built and
executed with:

```bash
python -m r2c_baselines.r2c_table234_completion_queue build
python -m r2c_baselines.r2c_table234_completion_queue worker
```

## Tests

```bash
python tests/run_tests.py
```

## Comparison scope

The matched comparison was frozen to methods with publicly released source
code that could be executed under the same model, partition, trace, seed,
round budget, and test-access order. More recent methods without a public
executable implementation at the protocol-freeze date were not added to the
formal comparison. Within that constraint, the paper compares against the
latest publicly available implementations that could be protocol-matched.

Third-party notices and preserved licenses are described in
`THIRD_PARTY.md`.
