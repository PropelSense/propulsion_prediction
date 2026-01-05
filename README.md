# Propulsion Power Prediction

This project predicts ship propulsion power consumption using machine learning. We've built several models (Random Forest, XGBoost, MLP neural networks, and Deep Ensembles) that take in operational parameters like speed, draft, wind, and current conditions to predict how much power the ship will need.

## Getting Started

Before cloning, you'll need Git LFS since the data files are large:

```bash
git lfs install
git clone https://github.com/PropelSense/propulsion_prediction.git
```

Then install the requirements:

```bash
pip install -r models/requirements.txt
```

## Project Structure

```
propulsion_prediction/
├── train_models.ipynb      # Train all models from scratch
├── generate_predictions.ipynb  # Load trained models and run predictions
├── data/                   # Training and test datasets
├── models/                 # All the model code lives here
├── checkpoints/            # Saved model weights and scalers
├── outputs/                # Prediction results and plots
└── utils/                  # Helper functions for evaluation
```

## The Two Main Notebooks

### `train_models.ipynb`

Run this notebook to train all four model types from scratch:

1. Loads and preprocesses the data (handles missing values, scales features)
2. Engineers useful features like speed cubed, wind magnitude, trim, etc.
3. Trains baseline models (Random Forest, XGBoost)
4. Trains neural networks (MLP and a 5-member Deep Ensemble)
5. Evaluates everything and saves the best checkpoints

The whole training pipeline takes maybe 10-15 minutes on a decent GPU. You can switch between synthetic and real data by changing `config.data.data_dir` near the top.

### `generate_predictions.ipynb`

Once you have trained models, this notebook loads them and generates predictions on new data. It:

1. Loads the saved models from `checkpoints/`
2. Applies the same preprocessing used during training
3. Generates predictions from all four models
4. Computes metrics (MAE, RMSE, MAPE, R²) for comparison
5. Creates visualizations showing prediction quality
6. Exports everything to CSV files in `outputs/predictions/`

## Current Results

On our test data, the models achieve around 13-15% MAPE (mean absolute percentage error):

| Model | In-Distribution R² | Out-of-Distribution R² |
|-------|-------------------|------------------------|
| Random Forest | 0.86 | 0.69 |
| XGBoost | 0.86 | 0.75 |
| MLP | 0.86 | 0.74 |
| Deep Ensemble | 0.86 | 0.74 |

XGBoost tends to be most robust when the data distribution shifts. The Deep Ensemble is nice because it also gives you uncertainty estimates.

## Data

We have two datasets in `data/`:

- **synthetic_data/** - Simulated data for development and testing
- **real_data/** - Actual ship operational data

Each contains:
- `train.csv` - Training data (~530k samples)
- `dev_in.csv` - In-distribution test set
- `dev_out.csv` - Out-of-distribution test set (for checking robustness)

## Features Used

The models use 19 features after engineering:

- Ship state: draft (fore/aft), trim, mean draft
- Speed: STW, STW², STW³ (power scales roughly with speed cubed)
- Environment: wind components, wind magnitude/angle, wave height
- Currents: current components, magnitude/angle
- Maintenance: time since last dry dock
- Interactions: speed × wind

## Need Help?

Check the README files in each subfolder for more details on what's in there. The code is reasonably well-commented, but feel free to dig into `models/` if you want to understand the implementation.
