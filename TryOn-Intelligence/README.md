# TryOn Intelligence – AR Shopping Analytics

An interactive analytics application exploring how Augmented Reality (AR) Virtual Try-On may shape customer confidence, omnichannel shopping behaviour and purchase intention in fashion, beauty and footwear retail.

## Research foundation

This portfolio application builds on the earlier research project, **AR Virtual Try-On as an Omni-Channel Strategy: Understanding Webrooming and Showrooming Behaviour Using TISM, MICMAC and DEMATEL Analysis**.

The framework includes six constructs and 21 five-point Likert items:

| Construct | Items | Focus |
|---|---:|---|
| Online Shopping Value | 1–3 | Convenience, comparison, information |
| Physical Store Value | 4–6 | Fit/quality checking, touch and feel, physical confidence |
| AR Experience Quality | 7–11 | Realism, accuracy, ease, responsiveness, interactivity |
| AR-Enabled Risk Reduction | 12–14 | Uncertainty, perceived risk, purchase confidence |
| Omnichannel Behaviour | 15–18 | Webrooming, showrooming, channel behaviour |
| Purchase Intention | 19–21 | Willingness to buy and use AR |

The theory lens combines S-O-R with Dual-Factor Theory. The earlier structural research also used TISM, MICMAC and DEMATEL.

## Application modules

- Executive Overview: construct performance and shopping journey mix.
- AR Experience Lab: item-level AR experience scores and reliability diagnostic.
- Omnichannel Journey: webrooming, showrooming and channel mix.
- Customer Insights: construct comparisons and correlations.
- Research Analytics: nine Pearson/Spearman hypothesis checks, multiple regression, confidence intervals and residual / collinearity diagnostics.
- Customer Segmentation: exploratory K-means clusters, PCA respondent map, silhouette score and segment profiles.
- Strategy Simulator: transparent illustrative what-if scenarios.
- Data Studio: completeness, construct diagnostics, previews and exports.

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Data format

Upload a CSV or Excel file with respondent-level rows and 21 five-point Likert item columns. The sidebar allows you to select the item columns in questionnaire order.

Mapping:
- Q1–Q3: Online Shopping Value
- Q4–Q6: Physical Store Value
- Q7–Q11: AR Experience Quality
- Q12–Q14: AR-Enabled Risk Reduction
- Q15–Q18: Omnichannel Behaviour
- Q19–Q21: Purchase Intention

Optional columns improve segmentation:
- Shopping Journey (e.g. Webrooming, Showrooming, Mixed, Online-first)
- Product Category (e.g. Fashion, Beauty, Footwear)
- AR Usage

## Data integrity and limitations

The built-in demo dataset is synthetically generated for interface exploration. It is **not** the original research survey and must not be cited as empirical findings. Upload the original respondent-level data for actual analysis.

The Strategy Simulator is an illustrative scenario index, not a causal model, forecast or operational guarantee. Research Analytics reports unadjusted p-values for nine hypotheses; consider multiple-testing correction and the small convenience sample before interpreting results. Regression treats multi-item Likert composites as approximately continuous, which should be disclosed. K-means segments are exploratory profiles, not verified market personas. Reliability and descriptive statistics should be interpreted with sample size, measurement quality and research design in mind. The earlier research report noted weak internal consistency and limited inferential support in the 70-response quantitative dataset; this application does not remove those limitations.

## Stack

Python · Streamlit · Pandas · NumPy · Plotly · SciPy · Statsmodels · scikit-learn

## Deployment

Designed for Streamlit Community Cloud. Set the app entry point to app.py.
