## sp500-pca-ml

**Interactive PCA analysis of S&P 500 stocks using Streamlit.**

This app:
- Loads S&P 500 constituents and sector weights.
- Uses historical daily prices for index members and the S&P 500 itself.
- Lets you explore:
  - Sector weights and top stocks by index weight.
  - Sector-wise cumulative returns and correlation.
  - Principal Component Analysis (PCA) of stock returns, including variance explained, loadings, and a PCA scatter plot.

### How to run the app locally

- **Install dependencies** (using your preferred tool, e.g. `uv` or `pip`):
  - `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `yfinance`, `selenium`
- **Run the Streamlit app** from the project root:

```bash
streamlit run app/Home.py
```

The side navigation in Streamlit lets you switch between:
- **Home** – PCA demo on simple 2D data.
- **S&P 500 Overview** – Sector weights, top stocks per sector, cumulative returns, and sector correlation.
- **PCA S&P 500** – PCA on the selected date range for the chosen S&P 500 universe.
