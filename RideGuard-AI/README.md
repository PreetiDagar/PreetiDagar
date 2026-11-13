# RideGuard AI 🚕
### Mobility Risk & Operations Intelligence

RideGuard AI is an end-to-end machine-learning portfolio application built around the NCR ride-booking analytics work. It combines booking-outcome prediction, route-level risk analysis, model diagnostics and an interactive scenario simulator.

> **Demo data notice:** The built-in demo dataset is synthetically generated for demonstration. It is not real NCR ride-booking history. Upload the original ride-booking CSV to analyse your actual data.

## Product features

- **Command Center:** booking outcome mix, completion rate, unsuccessful-outcome share and held-out model metrics.
- **Risk Lab:** estimate the outcome distribution for a proposed booking scenario.
- **Route Intelligence:** rank pickup–drop combinations by observed unsuccessful-outcome rate, with a minimum-volume filter.
- **Model Diagnostics:** confusion matrix, per-class precision/recall/F1, balanced accuracy, macro F1 and permutation importance.
- **Data Explorer:** inspect recognized outcomes and download cleaned labelled rows.

## Machine-learning approach

Two models are supported:

1. Logistic Regression with class balancing.
2. Random Forest with balanced subsampling.

The app uses a stratified train/test split and reports accuracy, balanced accuracy, macro F1 and per-class metrics. Model evaluation is not based on accuracy alone.

### Leakage-aware features

The model uses only booking-time fields:
- Booking hour
- Weekday
- Month
- Vehicle type
- Pickup location
- Drop location

It deliberately excludes final booking status, cancellation reasons, driver/customer ratings, arrival times and completion-time fields from predictors. The target is normalized into Completed, Cancelled, Incomplete and No Driver Found.

Permutation importance is calculated on a held-out test sample. It indicates model reliance on a feature, not causality.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Activate the environment
cd RideGuard-AI
pip install -r requirements.txt
streamlit run app.py
```

The app starts with a synthetic demo dataset. To analyse real data, select **Upload my ride data** in the sidebar and upload a CSV containing:

```text
Booking Status
Date
Time
Vehicle Type
Pickup Location
Drop Location
```

Other columns may be present; they are not used as model predictors.

## Deploy on Streamlit Community Cloud

1. This project currently lives in the `RideGuard-AI/` folder of the public `PreetiDagar/PreetiDagar` repository.
2. Open https://share.streamlit.io/ and connect your GitHub account.
3. Select the public repository `PreetiDagar/PreetiDagar`, branch `main`, and app entry point `RideGuard-AI/app.py`.
4. Deploy. No API key or secret is required for the current version.

## Project structure

```text
RideGuard-AI/
├── app.py
├── rideguard.py
├── requirements.txt
├── README.md
├── .gitignore
└── tests/
    └── test_rideguard.py
```

## Limitations

- Synthetic demo metrics are illustrative only and must not be presented as real-world model performance.
- Model performance depends on the quality, size, representativeness and class balance of the uploaded dataset.
- Historical associations do not establish causal effects.
- Risk scores are model estimates, not guarantees or dispatch recommendations.

## Author

**Preeti Dagar** · [GitHub](https://github.com/PreetiDagar) · [Portfolio](https://meek-mermaid-723af2.netlify.app/)
