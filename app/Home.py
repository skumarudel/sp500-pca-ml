import os
import sys

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Ensure project root is on sys.path when run from app directory (e.g. Streamlit Cloud)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import custom scripts for modularity and readability
from sp500_pca_ml.data_download import (
    saved_data,
    saved_stock_data,
    download_sp_data,
    download_stock_data,
)

from sp500_pca_ml.data_cleaning import remove_outliers_nsigma


from sp500_pca_ml.visualizations import (  # Custom Plotly-based visualization functions
    create_pie_plot,
    create_heatmap_plot,
    create_pca_variance_plot,
    create_demo_pca_plot,
)

# ======================== Streamlit Home Setup ========================
# ======================== Page Configuration ========================

st.set_page_config(layout="wide", page_title="Principal Component Analysis")

st.title("📊 PCA on S&P 500 Stocks")  # Page title

st.markdown(
    """
#### <span style='color:blue'>Principal Component Analysis (PCA)</span> is a powerful statistical technique used to reduce the dimensionality of complex datasets while preserving as much variability as possible. By transforming the original features into a new set of uncorrelated features called principal components, PCA helps identify the most significant patterns and trends in the data. In other words, it transforms a large set of variables into a smaller set of variables, while still containing most of the information from the larger set. This method is widely used in finance (and many other fields) to uncover hidden structures, reduce noise, and visualize high-dimensional market data—making it especially useful for analyzing stock movements and relationships within indices like the S&P 500 index.
""",
    unsafe_allow_html=True,
)


# Demo section
st.markdown("### <u><b><i>PCA Demo with Simple 2D Data:</i></b></u>", unsafe_allow_html=True)
st.markdown(
    "Use the **sidebar** to adjust the correlation between the two variables. "
    "This will dynamically update the dataset and show how the **principal components** respond to different correlation strengths."
)


# Sidebar filter section
st.sidebar.header("PCA Demo Data Input")  # Sidebar title
# Choose data source

# Slider input
corr_strength = st.sidebar.slider(
    label="Correlation Strength",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.2,
    help="Move the slider to simulate weak (0) to strong (10) correlation",
)

col1, col2 = st.columns(2)
fig1, fig2 = create_demo_pca_plot(corr_strength)
with col1:
    st.plotly_chart(fig1, use_container_width=True, key="demo_pca_chart")

with col2:
    st.plotly_chart(fig2, use_container_width=True, key="demo_bar_chart")


st.markdown("""Imagine you’re watching how two stocks move together. Sometimes they move in sync (high correlation), other times more randomly (low correlation). \
This demo shows how **Principal Component Analysis** finds the main "direction" in which the data varies—like drawing an arrow that points \
where most of the movement happens.

For example, when the correlation is set to **4.4** (adjustable via the sidebar), the **first principal component alone explains 92.4% of the total variance** in the data.  
This means we can project the entire dataset onto just this single component without losing much of the underlying information — making the data significantly easier to interpret without sacrificing its core structure.
""")

st.markdown(
    "### <u><b><i>🧠 PCA and the Nature of Reality: A Thought Experiment</i></b></u>",
    unsafe_allow_html=True,
)
st.markdown(
    """Our brains are constantly performing their own version of **PCA** — even if we don’t call it that.
In a world overflowing with sensory data — sights, sounds, emotions, and ideas — we instinctively try to simplify things and extract the **main patterns** that help us survive and make decisions.

Think about it:

- You walk into a busy room. You don't process **every single voice** or detail on every wall. Instead, your brain picks out what seems **most important**: a familiar face, a loudsound, or sudden motion.
- This is your mind performing a kind of **dimensionality reduction** — filtering out noise, and focusing on the **principal components of your reality**.""",
    unsafe_allow_html=True,
)

st.markdown("### <u><b><i>🤔 Why Use PCA on S&P 500 Stocks?</i></b></u>", unsafe_allow_html=True)
st.markdown("""
The S&P 500 includes **500 different companies**, each with its own stock price that moves daily. But in reality, **many of these stocks don’t move independently**. For example:

- Tech companies like **Apple, Microsoft, and Google** often rise or fall together.
- **Energy stocks** may move in sync due to changes in oil prices.
- Broader factors like **interest rate changes** can impact multiple sectors at once.

These **correlated movements** mean that the market’s behavior can often be explained using **fewer dimensions** than 500.

---

### ⚛️ A Physics Analogy: Symmetry Breaking

In physics, a perfectly symmetric system — like a uniformly magnetized material or a field in a vacuum — can spontaneously **break symmetry** and adopt a preferred direction when influenced by an external force.

Similarly, in financial markets, when correlations among stocks increase, the system “chooses” a dominant **direction of movement** — a kind of **market-wide trend**. PCA helps us detect this shift.

- For example, in times of crisis or euphoria, most stocks may move **together**, much like spins aligning in a ferromagnetic material.
- PCA identifies this dominant direction, showing us how collective behavior emerges from seemingly independent parts.
""")
