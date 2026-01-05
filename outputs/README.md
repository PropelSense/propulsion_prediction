# Outputs

This is where prediction results and visualizations get saved when you run `generate_predictions.ipynb`.

## predictions/

The main output folder. After running inference, you'll find:

**predictions_dev_in.csv** - Full predictions on the in-distribution test set. Contains actual values, predictions from each model, uncertainties from the ensemble, and errors.

**predictions_dev_out.csv** - Same thing for the out-of-distribution test set. Useful for checking how models behave when conditions change.

**metrics_summary.csv** - Quick comparison table with MAE, RMSE, MAPE, and R² for all models on both test sets. Good for a high-level view of performance.

**model_comparison.png** - Bar charts comparing MAE and RMSE across models. Shows in-distribution vs out-of-distribution side by side.

**predictions_scatter.png** - Predicted vs actual scatter plots for each model. Points should cluster around the diagonal if the model is good. 8 subplots total (4 models × 2 datasets).

**error_distributions.png** - Histograms of prediction errors. You want these centered around zero with small spread. Shows bias if the mean is off from zero.

**uncertainty_analysis.png** - Plots ensemble uncertainty against actual error. A good uncertainty estimate should correlate with error - high uncertainty samples should have higher errors. The correlation coefficient tells you how useful the uncertainty is.

**predictions_with_uncertainty.png** - Shows ensemble predictions with ±1σ and ±2σ bands for a subset of samples. Nice visual check that uncertainties are reasonable.

## Notes

These files get overwritten each time you run the prediction notebook. If you want to keep a specific run, copy the folder somewhere else or rename it.

The CSVs can get pretty large (18k+ rows each with 30+ columns). If you just want to check metrics, look at `metrics_summary.csv` first.

