## sp500-pca-ml

**Interactive PCA analysis of S&P 500 stocks using Streamlit.**

This app:
- Loads S&P 500 constituents and sector weights.
- Uses historical daily prices for index members and the S&P 500 itself.
- Lets you explore:
  - Sector weights and top stocks by index weight.
  - Sector-wise cumulative returns and correlation.
  - Principal Component Analysis (PCA) of stock returns, including variance explained, loadings, and a PCA scatter plot.

### Environment setup with `uv`

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd sp500-pca-ml

# Create and activate a Python environment (3.12 recommended)
uv venv
source .venv/bin/activate  # on macOS/Linux

# Install project dependencies
uv add streamlit pandas numpy plotly scikit-learn yfinance selenium
```

### How to run the app locally

- **Run the Streamlit app** from the project root:

```bash
streamlit run app/Home.py
```

The side navigation in Streamlit lets you switch between:
- **Home** – PCA demo on simple 2D data.
- **S&P 500 Overview** – Sector weights, top stocks per sector, cumulative returns, and sector correlation.
- **PCA S&P 500** – PCA on the selected date range for the chosen S&P 500 universe.
