# On the Tradeoffs of On-Device Generative Models in Federated Predictive Maintenance Systems

Release code for the paper *"On the Tradeoffs of On-Device Generative Models in Federated Predictive Maintenance Systems"*. It implements federated training of unsupervised generative models for time-series anomaly detection (TSAD) under full and partial (analysis/synthesis) parameter-sharing policies.

**Paper:** [arXiv:2605.07860](https://arxiv.org/abs/2605.07860)

## Models

| CLI name | Paper name | Analysis component (`enc`) | Synthesis component (`dec`) |
|---|---|---|---|
| `vae` | LSTM-VAE | Encoder | Decoder |
| `wgan_gp` | TAnoWGAN | Critic | Generator |
| `fedsw_tsad` | FedSW-TSAD | Discriminator | Generator (predictor stays local; shared only in full federation) |
| `ddpm` | TAnoDDPM | U-Net downsampling path | U-Net upsampling path |

## Experimental setups

- **Centralized**: single node with all training data (`main_independent.py --num_clients 1`).
- **Independent**: C=5 clients, local training only (`main_independent.py --num_clients 5`).
- **Federated**: C=5 clients with FedAvg aggregation of the full model (`--fed_type full`), the analysis component (`--fed_type enc`), or the synthesis component (`--fed_type dec`) (`main_federated.py`).

## Datasets

Datasets are **not distributed** with this repository:

- **ARAMIS** (primary PdM benchmark)
- **SWaT** (complementary benchmark)

## Usage

```bash
bash scripts/run_all_centralized.sh aramis
bash scripts/run_all_independent.sh aramis
bash scripts/run_all_federated.sh aramis
```