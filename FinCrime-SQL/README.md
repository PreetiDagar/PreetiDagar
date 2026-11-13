# 🛡️ FinCrime SQL
### Financial Crime Detection & Investigation Intelligence

An interactive SQL-powered transaction-monitoring lab built with **DuckDB, Python, Streamlit and Plotly**. Explore suspicious transaction patterns, inspect explainable detection rules, trace entity relationships and work through a lightweight investigation workflow.

<p align="center">
  <a href="https://preetidagar-czfmbvk6vg3yhaonetakar.streamlit.app/"><strong>🚀 Open the live application</strong></a>
  &nbsp; · &nbsp;
  <a href="https://github.com/PreetiDagar/PreetiDagar/tree/main/FinCrime-SQL"><strong>Browse the project</strong></a>
</p>

> **Demo data notice:** The public application currently uses fictional, synthetically generated customers, accounts and transactions. Detection outputs are illustrative review signals—not findings of misconduct, regulatory determinations or evidence of real-world model performance. Do not upload real customer or financial data to the public demo.

---

## ✨ Product experience

The interface uses a midnight-plum visual identity with orchid and lavender highlights, high-contrast typography, compact investigation panels and interactive visual analytics.

## 🔎 Application modules

| Module | What you can explore |
|---|---|
| **Command Center** | Transaction activity, monitored value, review signals, entity counts and a relationship intelligence view |
| **Transaction Explorer** | Search and filter transaction-level activity and inspect records |
| **Detection Lab** | Explore explainable SQL rule hits, severity, rationale and illustrative risk scores |
| **Network Map** | Inspect customer connections to counterparties and devices |
| **Investigation Desk** | Review alerts and manage demonstration investigation cases |
| **SQL Workbench** | Execute read-only SQL against the in-app DuckDB schema and inspect query results |
| **Data Quality** | Review data completeness, key integrity, duplicates and synthetic-scenario coverage |

## 🧠 SQL & analytics demonstrated

- Common Table Expressions (CTEs)
- Window functions and rolling time windows
- Multi-table joins and subqueries
- Conditional logic with `CASE WHEN`
- Aggregations, ranking and grouping
- Date/time analysis and transaction velocity checks
- Data-quality validation and query inspection

### Illustrative detection scenarios

- **Velocity bursts:** repeated transactions within a rolling time window.
- **Near-threshold cash activity:** clustered cash deposits in a configurable illustrative band.
- **Geographic inconsistency:** differences between merchant and IP country.
- **High-value activity:** transactions above an illustrative review threshold.

Rules are intentionally transparent so users can inspect the SQL logic and understand why a record enters the review queue. A rule hit can have legitimate explanations and requires human review.

## 🏗️ Technology stack

- **Application:** Python, Streamlit
- **SQL engine:** DuckDB
- **Data handling:** Pandas, NumPy
- **Visualisation:** Plotly
- **Data:** deterministic synthetic demo records generated in the application

## 🚀 Run locally

~~~bash
git clone https://github.com/PreetiDagar/PreetiDagar.git
cd PreetiDagar/FinCrime-SQL

python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
~~~

## ☁️ Streamlit Community Cloud deployment

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Connect your GitHub account and select `PreetiDagar/PreetiDagar`.
3. Choose the `main` branch.
4. Set the app file path to `FinCrime-SQL/app.py`.
5. Deploy and wait for dependency installation to complete.

The application creates its demo dataset in memory at startup and does not require external API credentials.

## 🗂️ Project structure

~~~text
FinCrime-SQL/
├── app.py              # Streamlit application and DuckDB workflows
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
~~~

## 🔐 Responsible-use & data notes

- The current public demo is synthetic and fictional.
- Synthetic patterns are planted to make selected rule behaviours inspectable.
- Risk scores and thresholds are illustrative, not calibrated for operational decisions.
- Geographic mismatches, transaction clusters and high values are not proof of criminal activity.
- Real deployments require secure data handling, governance, validation, threshold calibration, auditability and qualified human review.
- Do not submit confidential or personally identifiable financial information through the public application.

## 🧭 Future enhancements

- Configurable detection-rule management and version history.
- Persistent investigation notes and reviewer audit trails.
- Optional PostgreSQL integration for separately configured environments.
- Automated tests for rule logic and SQL query safety.
- Benchmarking against labelled synthetic scenarios.

---

**Built as a portfolio project demonstrating SQL analytics, transaction monitoring concepts, data visualisation and investigation-oriented application design.**
