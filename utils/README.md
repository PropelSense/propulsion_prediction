# Utils

Helper functions that don't fit neatly into the models folder. Mostly evaluation and plotting stuff.

## Files

**assessment.py** - Performance metrics and uncertainty evaluation. The main function here is `calc_uncertainty_regection_curve()` which computes how error changes as you reject high-uncertainty predictions. Also has `get_performance_metric()` for standard regression metrics. Some of this code is adapted from the Yandex Shifts benchmark.

**auxiliary_functions.py** - Miscellaneous helper functions. Data manipulation, formatting, that kind of thing.

**plot_utils.py** - Plotting helpers for consistent styling across visualizations. If you want to change how the plots look, this is where to do it.

**__init__.py** - Just makes this a Python package so you can do `from utils import ...`

## Usage

These get imported in the notebooks:

```python
from assessment import get_performance_metric
```

The uncertainty rejection curve is particularly useful - it tells you how much you can improve predictions by filtering out samples where the model is uncertain. If your uncertainty estimates are well-calibrated, rejecting the top 20% most uncertain predictions should noticeably reduce error on the remaining 80%.

## Why separate from models/?

Kept these here to avoid cluttering the core model code. The stuff in `models/` is focused on training and inference. This folder is more about analyzing results after the fact.

