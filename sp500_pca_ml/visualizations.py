import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_pie_plot(spdata):
    """
    Create a pie plot to visualize the sector wise contribution of
    stocks to the S and P index fund
    It will take s and p data frame as input
    and return pie chart

    """
    sector_counts = spdata.groupby("GICS Sector")["Weight"].sum().sort_values(ascending=False)
    return px.pie(
        values=sector_counts.values,
        names=sector_counts.index,
        title="S&P 500 Sector-wise Contribution",
        hole=0.3,  # for a donut-style chart
    )


def create_heatmap_plot(sector_corr):
    """
    It will create a correlation map between all sectors in S and P fund
    Better insight to check which sectors are correlated
    """

    return px.imshow(
        sector_corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Sector wise Correlation map",
    )


def create_pca_variance_plot(explained_variance):
    x = np.arange(1, len(explained_variance) + 1)
    cumulative_variance = np.cumsum(explained_variance)
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Explained variance ratio (bar)
    fig.add_trace(
        go.Bar(
            x=x,
            y=explained_variance,
            name="Explained Variance Ratio",
            marker_color="steelblue",
        ),
        secondary_y=False,
    )

    # Cumulative explained variance (line)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative_variance,
            mode="lines+markers",
            name="Cumulative Explained Variance",
            line=dict(color="firebrick"),
        ),
        secondary_y=True,
    )

    # Layout settings
    fig.update_layout(
        title_text="Explained and Cumulative Variance",
        xaxis_title="Principal Component",
        legend=dict(x=0.5, y=1.15, orientation="h", xanchor="center"),
        template="plotly_dark",  # or 'plotly' if you're not using a dark theme
        height=500,
    )

    fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)

    return fig


def create_demo_pca_plot(correlation_strength):
    import numpy as np
    from sklearn.decomposition import PCA

    np.random.seed(20)
    mean = [0, 0]
    cov = [[5, correlation_strength], [correlation_strength, 5]]  # Correlated 2D data
    X = np.random.multivariate_normal(mean, cov, 200)

    # Apply PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    # Principal component vectors
    vectors = []
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        vectors.append(v)

    # Create plotly figure
    fig = go.Figure()

    # Original data scatter plot
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            name="Original Data",
            marker=dict(color="blue", opacity=0.5),
        )
    )

    # PCA component arrows
    for i, v in enumerate(vectors):
        fig.add_trace(
            go.Scatter(
                x=[0, v[0]],
                y=[0, v[1]],
                mode="lines+markers+text",
                line=dict(color="red", width=4),
                marker=dict(size=[0, 5]),
                name=f"PC {i + 1}",
                text=[None, f"PC{i + 1}"],
                textposition="top right",
            )
        )

        fig.update_layout(
            title="2D PCA Demo",
            xaxis_title="X1",
            yaxis_title="X2",
            showlegend=True,
            width=600,
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
        )

    # Explained variance ratio (bar)
    # Plot explained variance ratio
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=[f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
            name="Explained Variance Ratio",
            marker_color="#FF474C",
            width=[0.2] * len(pca.explained_variance_ratio_),
        )
    )

    fig2.update_layout(
        title="Contribution of each component to the data’s variance",
        xaxis_title="Principal Components",
        yaxis_title="Variance Ratio",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        width=200,
        height=500,
    )

    return fig, fig2


def create_sp_top5_barplot(dataframe):
    df = dataframe[["Company", "Weight"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(df["Company"].to_numpy()),
            y=[value * 100 for value in df["Weight"].to_list()],
            name="Percentage Contribution",
            marker_color="#FFA3DD",
            width=[0.5] * len(df),
        )
    )

    fig.update_layout(
        title="Sectorwise Contribution of Top 10 stocks to S&P index",
        xaxis_title="",
        yaxis_title="Percentage Contribution",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )

    return fig


def cumulative_return_sectorwise(spdata, stock_data, top_value):
    fig = go.Figure()
    sectors = list(spdata["GICS Sector"].unique())

    for sector in sectors:
        df_sector = (
            (spdata[spdata["GICS Sector"] == sector])
            .sort_values(by="Weight", ascending=False)
            .head(top_value)
        )
        symbol_list = df_sector.Symbol.to_list()
        weights = df_sector["Weight"]
        weights = weights / weights.sum()

        filtered_stockdata = stock_data[symbol_list]
        returns = filtered_stockdata.pct_change().dropna()
        weighted_daily_return = pd.DataFrame(
            {"weighted return": (returns.to_numpy() @ weights.to_numpy())},
            index=returns.index,
        )

        sector_cum_returns = (1 + weighted_daily_return).cumprod() - 1

        fig.add_trace(
            go.Scatter(
                x=sector_cum_returns.index,
                y=sector_cum_returns["weighted return"],
                mode="lines",
                name=sector,
            )
        )

    # Add S&P 500 index line
    spindex = stock_data["^GSPC"].pct_change().dropna()
    spindex_cum_returns = (1 + spindex).cumprod() - 1

    fig.add_trace(
        go.Scatter(
            x=spindex_cum_returns.index,
            y=spindex_cum_returns,
            mode="lines",
            name="S&P 500",
            line=dict(dash="solid", color="#282C35", width=5),
        )
    )

    fig.update_layout(
        title="Cumulative Returns by Sector (Top {} Stocks per Sector)".format(top_value),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend_title="Sector",
        height=600,
        template="plotly_white",
    )

    return fig


def create_pca_scatter_plot(spdata, pca_components, component1="PC1", component2="PC2"):
    """
    Create a scatter plot of the PCA components with sector-wise coloring.
    """
    # Ensure pca_components is a DataFrame
    if not isinstance(pca_components, pd.DataFrame):
        raise ValueError("pca_components must be a pandas DataFrame")

    # Add sector information to the PCA components

    sector_map = {spdata["Symbol"][i]: spdata["GICS Sector"][i] for i in range(len(spdata))}
    t_pca_components = pca_components.T
    t_pca_components["Sector"] = t_pca_components.index.map(sector_map)

    fig = px.scatter(
        t_pca_components,
        x=component1,
        y=component2,
        color="Sector",
        title="PCA Scatter Plot of S&P 500 Stocks",
        labels={component1, component2},
        hover_data=["Sector"],
        template="plotly_white",
    )

    return fig
