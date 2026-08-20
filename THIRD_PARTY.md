# Third-party implementation references

The `upstream/` directory preserves read-only source snapshots used to inspect
the semantics of selected baseline adapters. Each snapshot retains its own
license file and remains governed by that upstream license.

| Directory | Upstream project | Pinned revision recorded by the harness |
|---|---|---|
| `upstream/fedau` | IBM FedAU | `612814c9791a1e41a8a1b123616af52377a224b9` |
| `upstream/f3ast` | Google Research federated optimization / F3AST | `27b4e33adffea94a3fe53a5600fe5498f4cd3d5d` |
| `upstream/FedAWE` | Official FedAWE implementation | `e0b8538adc95dcbcb63574729594ad4605df969b` |
| `upstream/Oort` | SymbioticLab Oort | `05a3aa1677a10f8e621055b1626ef82e73d09759` |

FedAvg, Power-of-Choice, and TiFL are implemented as protocol-matched adapters
from their published algorithm descriptions. The source provenance identifier
for every executed method is recorded in `r2c_baselines/run.py` and written to
each run manifest.
