import pandas as pd


# Assuming `df` contains time series of different stocks, with datetime index
def remove_outliers_nsigma(df, n):
    """
    Remove any date where return is 3 sigma away
    from the mean value of standrized return
    if it is 3 sigma away then, replace it with NaN
    then drop all rows where there are NaN values
    Returns:
        dataframe: cleaned dataframe with outliers removed
    """
    df_cleaned = df.copy()
    for col in df.columns:
        series = df[col]
        mean = series.mean()
        std = series.std()
        # Keep values within ±3σ, set others to NaN
        df_cleaned[col] = series.where((series > mean - n * std) & (series < mean + n * std))
    return df_cleaned.dropna()
