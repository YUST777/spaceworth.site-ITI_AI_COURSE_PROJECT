---
license: mit
library_name: joblib
pipeline_tag: tabular-regression
tags:
  - house-price-prediction
  - lightgbm
  - catboost
  - pytorch
  - ensemble
---

# ITI House Price Ensemble

The final house-price regression model for the ITI project. It combines LightGBM, CatBoost, and three PyTorch entity-embedding networks.

## Validation result

- Held-out R²: **0.9064493141**
- Accuracy expressed as R² percentage: **90.64%**
- MAE: **₹1,535,857.51**
- RMSE: **₹2,883,442.92**

The serialized model is stored in 2 MiB files under `house_price.pkl.parts/` so it can be uploaded reliably from a time-limited client. Concatenate them in order to reconstruct `house_price.pkl`; the complete evaluation report is in `model_metrics.json`.

```bash
git clone https://huggingface.co/duck233/iti-house-price-model
cat iti-house-price-model/house_price.pkl.parts/part-* > house_price.pkl
```

The model intentionally excludes the dataset's source price-per-square-foot field and price-containing description text to avoid target leakage.
