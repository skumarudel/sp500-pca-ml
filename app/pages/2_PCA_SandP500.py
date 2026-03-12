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

from sp500_pca_ml.visualizations import (
    create_pie_plot,
    create_heatmap_plot,
    create_pca_variance_plot,
    create_sp_top5_barplot,
    cumulative_return_sectorwise,
    create_pca_scatter_plot,
)

from sp500_pca_ml.utils import create_sectorwise_corr

from sp500_pca_ml.pca_analysis import run_PCA

# ======================== Streamlit Page Setup ========================
st.set_page_config(page_title="S and P 500 index Overview", layout="wide")
st.title("📈 PCA on Real S&P 500 Data for Investment Insights")  # Dashboard title
st.markdown(
    """In this section, we apply Principal Component Analysis (PCA) to historical stock price data from companies in the S&P 500 index. By reducing dimensionality, PCA helps us identify underlying patterns and common movements in the market, allowing us to group stocks with similar behavior and construct data-driven portfolios. There are various sources of risk in an asset portfolio, including market risk, sector risk, and asset-specific risk. PCA helps identify and quantify these risks by breaking down the returns of the portfolio into components that explain the maximum variance. The first few principal components usually capture most of the variance and they can be analyzed to understand the major sources of risk in the portfolio. We analyze the loadings on the top principal components to infer stock weights, explore sector-based trends to enhance investment decision-making. This analysis serves as a foundation for building interpretable and robust stock selection strategies."""
)

col1, spacer, col2 = st.columns([3, 0.5, 3])

if "spdata" in st.session_state and "stockdata" in st.session_state:
    with col1:
        num_components = st.slider(
            "Variance explained",
            0.0,
            1.0,
            0.7,
            0.1,
            help="Select the variance explained by the PCA components",
        )
        normalize = st.checkbox("Normalize returns", value=True)
        # Load data from session state
        stockdata = st.session_state["stockdata"]
        spdata = st.session_state["spdata"]
        returns, pca, returns_pca = run_PCA(stockdata, num_components, normalize)
        if pca is not None and returns_pca is not None:
            # st.success("PCA analysis completed successfully!")
            fig = create_pca_variance_plot(pca.explained_variance_ratio_)
            st.plotly_chart(fig, use_container_width=True, key="PCA variance plot")

    with col2:
        pca.components_ = pd.DataFrame(
            pca.components_,
            index=[f"PC{i + 1}" for i in range(pca.n_components_)],
            columns=returns.columns,
        )
        loadings = pca.components_.T
        pcs_to_show = st.multiselect(
            "Select Principal Components", loadings.columns.tolist(), default=["PC1", "PC2"]
        )
        st.markdown("### Loadings (factor-exposure) of Selected Principal Components")
        st.markdown(
            "The loadings indicate how much each asset in the portfolio is exposed to a particular principal component (factor)."
            "In other words, The exposure of each asset to each of these factors, lets say "
            "('PC1', 'PC2') represents how much the returns of that asset are influenced by changes in those factors."
        )
        st.dataframe(loadings[pcs_to_show].round(3), use_container_width=True)
else:
    st.warning(
        "⚠️ Data not found! Please go back to the previous page and load the data before proceeding."
    )

st.markdown(
    "### Scatter plot of Principal components"
)  # Section title for sector-wise correlation heatmap
col1, spacer, col2 = st.columns([3, 0.5, 3])

with col1:
    columns = list(pca.components_.T.columns)

    component1 = st.selectbox("Select X-axis column", columns)
    component2 = st.selectbox("Select Y-axis column", columns, index=1)

    st.plotly_chart(
        create_pca_scatter_plot(
            st.session_state["spdata"], pca.components_, component1, component2
        ),
        use_container_width=True,
        key="PCA scatter plot",
    )
