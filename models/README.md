# Models

This folder contains all the model implementations and training logic.

## Files

**config.py** - Configuration classes for the whole project. If you need to change hyperparameters, hidden layer sizes, learning rates, etc., this is the place. The `default_config` object gets imported everywhere.

**data_module.py** - Handles all the data loading and preprocessing. The `PropulsionDataModule` class loads CSVs, applies feature engineering, fits scalers on training data only (important!), and serves up batches. Also has `FeatureEngineer`, `DataScaler`, and `TargetScaler` classes that can be saved/loaded for inference.

**baseline_models.py** - Tree-based models. Contains wrappers for Random Forest and XGBoost with sensible defaults. Also has a dead-simple `MeanBaseline` that just predicts the average (useful sanity check - if your real model can't beat this, something's wrong).

**neural_models.py** - PyTorch neural networks. The `MLP` class is a straightforward multi-layer perceptron. `GaussianMLP` outputs both mean and variance for uncertainty estimation. `NeuralNetworkTrainer` handles the training loop, early stopping, checkpointing, etc.

**uncertainty.py** - Uncertainty quantification stuff. `DeepEnsemble` trains multiple MLPs with different random seeds and uses their disagreement as uncertainty. `MCDropout` keeps dropout on at inference time for uncertainty via multiple forward passes. There's also some utility functions for analyzing how well the uncertainties are calibrated.

**train.py** - Standalone training script if you prefer command line over notebooks. Not really used since the notebooks are more convenient for experimentation.

**evaluate.py** - Evaluation utilities. Functions for computing metrics, generating plots, comparing models, etc.

**requirements.txt** - Python dependencies. Run `pip install -r requirements.txt` before anything else.

**check_gpu.py** - Quick script to verify PyTorch can see your GPU. Run this if things are slow and you're not sure if CUDA is working.

## How it all fits together

The typical flow is:

1. `data_module.py` loads raw data, applies `FeatureEngineer`, fits scalers
2. Models from `baseline_models.py` or `neural_models.py` get trained
3. Trained models and scalers get saved to `../checkpoints/`
4. At inference time, load everything back and call `.predict()`

The notebooks in the parent directory orchestrate all of this we don't really need to run these files directly.

