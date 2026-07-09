![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

# Transformer Diffusion Hydrology

Transformer-based diffusion models for the simulation and reconstruction of hydro-meteorological time-series.

*This work builds on [CSDI](https://github.com/ermongroup/csdi) (Tashiro et al.) and was developed as part of a post-doctoral research project at INRAE.*

## Overview

Modeling hydro-meteorological time-series from limited observations is a key challenge for monitoring hydro-systems and water resources, and for flood or drought risk assessment. High process variability and sparse measurements often limit the accuracy of traditional statistical approaches.

This repository implements transformer-based diffusion models for the joint modeling of water quantity and quality, applied to six sites across three adjacent headwater catchments in North-East France. The models are evaluated against several established time-series baselines on two tasks:

- **Imputation** of incomplete time-series
- **Forecasting** of future hydrological conditions

## Repository structure

```
project/
├── data/                 # Data (see Data section)
├── notebooks/            # Example notebooks
├── configs/               # Model configuration / hyperparameter files
├── saved_weights/        # Pretrained model weights
├── models/
│   ├── wrapper_$BASELINENAME$.py   # Wrappers for external baseline libraries
│   ├── $MODELNAME$_architecture.py # Model architecture
│   └── $MODELNAME$_main.py         # Main model interface
├── utils.py               # Training loop, metrics, and visualization tools
├── dataset_ope.py         # PyTorch dataset for OPE data
├── dataset_synthetic.py   # PyTorch dataset for synthetic data
├── install.sh             # Dependency installation script
└── requirements.txt       # Python dependencies
```

## Installation

```bash
git clone https://github.com/Pumafi/transformer_diffusion_hydrology.git
cd transformer_diffusion_hydrology
bash install.sh
```

**Tested with:** Python 3.11.10, PyTorch 2.12.1+cu130

## Data

Raw data is not distributed with this repository. Two options are available:

- Generate synthetic data with `notebooks/synthetic_data_generation.ipynb`
- Use your own data, preprocessed as a CSV file with a regular time step

Our pretrained models were trained on OPE monitoring data provided by ANDRA (French Nuclear Waste Agency).

Expected input format:

```python
observed_ndarray = np.genfromtxt(
    "data/ope_andra/hydrology_sagd_training_data_50_hte.csv",
    delimiter=",", usemask=True, filling_values=0
)
```

An optional covariate file (e.g., SAFRAN meteorological data) can be supplied for the augmented model, covering the observation window plus the forecast horizon.

## Models

| Model | Description |
|---|---|
| U-Net | Diffusion backbone baseline |
| CSDI | Original implementation |
| CSDI Custom | Adapted for water level and quality data |
| CSDI Custom + Covariates | Augmented with weather covariates (SAFRAN reanalysis Météo France) |
| TSMixer (Probabilistic / Deterministic) | Darts implementation |
| N-HiTS | Darts implementation |

## Usage

```python
import numpy as np
import torch
import yaml
import sys, os

sys.path.append("..")
sys.path.append(os.path.abspath("../models"))
from models.custom_main import Custom_CSDI_WaterQual
from utils import *
from dataset_ope import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and normalize data
observed_ndarray = np.genfromtxt(
    "../data/ope_andra/hydrology_sagd_training_data_50_hte.csv",
    delimiter=",", usemask=True, filling_values=0
)
observed_ndarray = observed_ndarray - observed_ndarray.mean() / observed_ndarray.std()
observed_data, observed_mask = observed_ndarray.data, observed_ndarray.mask
data_dim = observed_data.shape[-1]

# Observation window and forecast horizon
nb_of_steps_used_for_obs = 96
nb_of_steps_to_predict = 90
observed_data = observed_data[-nb_of_steps_used_for_obs:]
observed_mask = observed_mask[-nb_of_steps_used_for_obs:]

data = np.concat([observed_data, np.zeros((nb_of_steps_to_predict, data_dim))], axis=0)
mask = np.concat([observed_mask, np.zeros((nb_of_steps_to_predict, data_dim))], axis=0)
timestamps = np.arange(data.shape[0]) * 1.0

data = np.expand_dims(data, axis=0)
mask = np.expand_dims(mask, axis=0)
timestamps = np.expand_dims(timestamps, axis=0)

# Load config and model
with open("../config/water_quality_imputation.yaml", "r") as f:
    config = yaml.safe_load(f)

model = Custom_CSDI_WaterQual(config, device, target_dim=data_dim).to(device)
model.load_state_dict(torch.load("../saved_weights/custom_model_weights.pt", weights_only=True))

data = torch.Tensor(data).to(device, dtype=torch.float32)
mask = torch.Tensor(mask).to(device, dtype=torch.float32)
timestamps = torch.Tensor(timestamps).to(device)

# Generate samples
generated = model.impute(data, mask, n_samples=30, timestamps=timestamps)
```

Additional examples (training, forecasting, baseline models) are available under `notebooks/`.

## Configuration

Model and training hyperparameters are defined in YAML files under `configs/` (see `config/water_quality_imputation.yaml` for an example).

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{transformer_diffusion_hydrology,
  title   = {Transformer-based diffusion models for hydrological time-series simulation and reconstruction},
  author  = {},
  journal = {},
  year    = {}
}
```

## License

See `LICENSE` for details.

## Acknowledgements

This work was conducted as part of a post-doctoral research project at INRAE. We thank the Geolearning Chair for funding and ANDRA for providing the data.