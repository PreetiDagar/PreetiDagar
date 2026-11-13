# FinCrime SQL — Financial Crime Detection & Investigation Lab

A portfolio-grade SQL analytics application demonstrating transaction monitoring, explainable rule-based alerts, customer risk profiling, investigation workflows and an interactive SQL workbench.

> **Important:** The public demo uses entirely synthetic transactions and fictional entities. It is an educational portfolio project, not a production AML system, legal/compliance advice, or a substitute for regulated transaction-monitoring controls. Alert rules and thresholds are illustrative and require calibration, governance and human investigation in real deployments.

## Project modules

- **Executive Command Center:** transaction volume, value, alert counts, severity mix and trend analysis.
- **Transaction Explorer:** filter and inspect transaction-level activity.
- **SQL Workbench:** run read-only SQL queries against the demo DuckDB database, inspect results and export CSV.
- **Detection Lab:** inspect explainable rule hits for transaction velocity, possible structuring, high-value activity and geographic inconsistency.
- **Entity Intelligence:** customer-level exposure, transaction patterns and risk indicators.
- **Investigation Desk:** create and manage demonstration cases from alerts.
- **Controls & Data Quality:** inspect coverage, nulls, duplicate identifiers and synthetic scenario validation.

## SQL concepts demonstrated

CTEs, window functions, conditional aggregation, date/time logic, joins, subqueries, CASE WHEN, ranking, grouping, data-quality checks and query-plan inspection.

## Architecture

- Python + Streamlit for the application layer.
- DuckDB as an embedded analytical SQL engine.
- Pandas/NumPy for deterministic synthetic demo-data generation.
- Plotly for interactive visual analytics.
- SQL schema and detection examples are provided in sql/.

## Run locally

~~~bash
cd FinCrime-SQL
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
~~~

## Deploy on Streamlit Community Cloud

1. Select repository PreetiDagar/PreetiDagar.
2. Select branch main.
3. Set **Main file path** to FinCrime-SQL/app.py.
4. Deploy. Streamlit installs packages from FinCrime-SQL/requirements.txt when the app directory is selected as the entry-point directory; if dependency resolution does not find it, use a root-level requirements file or move the app to its own repository.
5. The app generates deterministic synthetic data at startup; no external credentials or private data are needed.

## Data and methodology notes

- The demo dataset is generated in code with a fixed random seed for repeatability.
- Rule hits are deterministic and explainable. A hit is a review signal, not proof of illicit activity.
- The demo contains intentionally planted synthetic patterns to make the detection workflow inspectable.
- Do not upload real customer, account or transaction data to the public demo.
- Thresholds should never be interpreted as regulatory limits.

## Roadmap

- Add PostgreSQL connector for a separately configured database.
- Add configurable rule versioning and audit history.
- Add role-based case assignment and reviewer notes.
- Add explainable model benchmarking against synthetic scenario labels.
- Add automated unit tests for rule logic and query safety.
