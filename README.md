# Structure-informed Risk Minimization for Robust Ensemble Learning

## Abstract
Ensemble learning is a powerful approach for improving generalization under distribution shifts, but its effectiveness heavily depends on how individual models are combined. Existing methods often optimize ensemble weights based on validation data, which may not represent unseen test distributions, leading to suboptimal performance in out-of-distribution (OoD) settings. Inspired by Distributionally Robust Optimization (DRO), we propose Structure-informed Risk Minimization (SRM), a principled framework that learns robust ensemble weights without access to test data. Unlike standard DRO, which defines uncertainty sets based on divergence metrics alone, SRM incorporates structural information of training distributions, ensuring that the uncertainty set aligns with plausible real-world shifts. This approach mitigates the over-pessimism of traditional worst-case optimization while maintaining robustness. We introduce a computationally efficient optimization algorithm with theoretical guarantees and demonstrate that SRM achieves superior OoD generalization compared to existing ensemble combination strategies across diverse benchmarks.

## Installation
Our code is adapted from the open-source [wilds](https://github.com/p-lambda/wilds/tree/main) codebase. Please refer to their guidelines for installation.

### Datasets
We use FMoW-WILDS dataset for experiments.

You can download the dataset with the following command:
```
python wilds/download_datasets.py --root_dir ${data_root} --datasets fmow
```

### Ensemble Training
Please follow [DiWA](https://github.com/alexrame/diwa) to train indivisual models and get the results for uniform ensemble and greedy selection baselines.

### Results on FMoW-WILDS
To reproduce the redults on FMoW-WILDS, please run:
```
./experiment.sh
```
