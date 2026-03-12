import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from itertools import chain


def create_sectorwise_corr(spdata, stockdata, top_value):
    """
    This is create sector wise correlation
    by combing stocks in same sectors
    """
    sector_map = spdata.set_index("Symbol")["GICS Sector"].to_dict()
    sector_returns = pd.DataFrame()
    for sector in spdata["GICS Sector"].unique():
        df_sector = (
            (spdata[spdata["GICS Sector"] == sector])
            .sort_values(by="Weight", ascending=False)
            .head(top_value)
        )
        sector_tickers = df_sector.Symbol.to_list()
        sector_data = stockdata[sector_tickers]
        sector_returns[sector] = sector_data.mean(axis=1)
    # Step 7: Correlation matrix
    sector_corr = sector_returns.corr()
    return sector_corr
