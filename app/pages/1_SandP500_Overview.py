# ====================== Import RequiredA Libraries ===AA===================
import streamlit as st
import pandas as pd
import datetime

# Import custom scripts for modularity and readability
from sp500_pca_ml.data_download import (
    download_sp_data,
    download_stock_data,
    saved_data,
    saved_stock_data,
)

from sp500_pca_ml.visualizations import (  # Custom Plotly-based visualization functions
    create_pie_plot,
    create_heatmap_plot,
    create_pca_variance_plot,
    create_sp_top5_barplot,
    cumulative_return_sectorwise,
)

from sp500_pca_ml.utils import create_sectorwise_corr

# ======================== Streamlit Page Setup ========================
st.set_page_config(page_title="S and P 500 index Overview", layout="wide")
st.title("📈 S & P 500 index Overview")  # Dashboard title
st.markdown(
    """The S&P 500 Index (Standard & Poor’s 500) is a stock market index that tracks the performance of 500 of the largest publicly traded companies in the United States. It represents a broad cross-section of the U.S. economy, covering sectors like technology, healthcare, finance, and consumer goods. Widely regarded as a benchmark for the overall U.S. stock market, the S&P 500 is market-capitalization weighted, meaning companies with larger market values have a greater influence on the index's performance. Investors and analysts often use it to gauge the health and trends of the U.S. economy."""
)


# Choose data source
# Sidebar filter section
st.sidebar.header("Stock Data Input")  # Sidebar title
# Choose data source

data_source = st.sidebar.radio(
    "Select Stock Source", ("Use S&P 500 Stocks", "Enter Custom Tickers separated")
)

# Depending on selection, set tickers
if data_source == "Use S&P 500 Stocks":
    spdata = saved_data()
    st.session_state["spdata"] = spdata
    tickers = spdata["Symbol"].to_list()
    st.sidebar.success(f"Using {len(tickers)} S&P 500 tickers")

else:
    ticker_input = st.sidebar.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    spdata = saved_data()
    filtered_spdata = spdata[spdata["Symbol"].isin(tickers)]
    filtered_spdata = filtered_spdata.reset_index(drop=True)
    st.session_state["spdata"] = filtered_spdata
    if tickers:
        spdata = saved_data()
        filtered_spdata = spdata[spdata["Symbol"].isin(tickers)]
        filtered_spdata = filtered_spdata.reset_index(drop=True)
        st.session_state["spdata"] = filtered_spdata
        st.sidebar.success(f"Using {len(tickers)} Custom tickers")
    else:
        st.sidebar.warning("No tickers entered. Please enter valid stock symbols.")


# Sidebar control for how many top-weight stocks per sector to use
top_stocks_value = st.sidebar.number_input(
    "Top stocks by weight per sector", min_value=1, max_value=10, step=1, value=5
)


col1, col2 = st.columns(2)

with col1:
    if "spdata" in st.session_state:
        st.plotly_chart(
            create_pie_plot(st.session_state["spdata"]), use_container_width=True, key="pi_chart"
        )

with col2:
    if "spdata" in st.session_state:
        options = list(st.session_state["spdata"]["GICS Sector"].unique())
        selected_option = st.selectbox("Select a sector:", options)

        top_stocks = (
            st.session_state["spdata"][
                st.session_state["spdata"]["GICS Sector"] == selected_option
            ]
            .sort_values(by="Weight", ascending=False)
            .head(top_stocks_value)
        )

        st.plotly_chart(
            create_sp_top5_barplot(top_stocks),
            use_container_width=True,
            key="sectorwise barchart",
        )


if "spdata" in st.session_state:
    st.markdown("### Sectorwise Cumulative return and correlation plots")
    st.markdown("""
    **Sector-wise Cumulative Return and Correlation Plots** provide valuable insights into the behavior of different sectors within the S&P 500 index.  
    - **Cumulative return plots** help identify which sectors have outperformed or underperformed over time, offering a clearer picture of long-term trends and investment opportunties.
    - **Correlation plots** reveal how sectors move in relation to one another, which is crucial for diversification and risk management.  
    Together, these visualizations help investors understand sector dynamics, reduce portfolio risk, and make more informed allocation decisions within the S&P 500.
    """)

    # Date inputs constrained to dataset range: 2023-05-01 to 2025-05-22
    min_date = datetime.date(2023, 5, 1)
    max_date = datetime.date(2025, 5, 22)
    default_start_date = min_date
    # choose a reasonable default window inside the allowed range
    default_end_date = datetime.date(2023, 9, 1)
    start_date = st.sidebar.date_input(
        "Start date", value=default_start_date, min_value=min_date, max_value=max_date
    )
    end_date = st.sidebar.date_input(
        "End date", value=default_end_date, min_value=min_date, max_value=max_date
    )

    # Validation check
    if end_date <= start_date:
        st.error("End date must be after start date.")
    else:
        st.session_state["stockdata"] = saved_stock_data(start_date, end_date)

# col1, col2 = st.columns(2)
col1, col2 = st.columns([4, 2])
with col1:
    st.plotly_chart(
        cumulative_return_sectorwise(
            st.session_state["spdata"], st.session_state["stockdata"], top_stocks_value
        ),
        use_container_width=True,
        key="sectorwise cumulative return",
    )

with col2:
    # st.markdown("### Sectorwise Correlation plot")

    st.plotly_chart(
        create_heatmap_plot(
            create_sectorwise_corr(
                st.session_state["spdata"], st.session_state["stockdata"], top_stocks_value
            )
        ),
        use_container_width=True,
        key="sectorwise correlation map",
    )
