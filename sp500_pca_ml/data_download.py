import yfinance as yf
import pandas as pd
import time
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sp500_pca_ml.data_cleaning import remove_outliers_nsigma
from sp500_pca_ml.paths import DATA_DIR


@st.cache_data
def saved_data():
    """
    This function will load the saved S and P data
    from the data directory and return it as a pandas DataFrame.

    Returns:
        data frame: pandas DataFrame containing the S and P 500 data
    """
    df = pd.read_csv(str(DATA_DIR) + "/sandpdata.csv", index_col=0)
    df.index.name = "index"
    return df


@st.cache_data
def saved_stock_data(start_date, end_date):
    """
    This function will load the saved stock data between two dates
    from the data directory and return it as a pandas DataFrame.

    Args:
        start_date (string): string representing the start date in 'YYYY-MM-DD' format
        end_date (string): string representing the end date in 'YYYY-MM-DD' format

    Returns:
        dataframe: pandas DataFrame containing the stock data
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = pd.read_csv(
        str(DATA_DIR) + "/stock_date_top10_2023-04-30_2025-05-22.csv", index_col="Date"
    )
    df.index = pd.to_datetime(df.index)
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df[mask]
    return filtered_df


@st.cache_data(show_spinner=True)
def download_sp_data():
    """
    This function will download the S and P 500 data from
    slickcharts and wikipedia using selenium webdriver.
    It will return a pandas DataFrame containing the S and P 500 data

    Returns:
        _type_: pandas DataFrame containing the S and P 500 data on all the stocks in the index
    """

    service = Service()
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(service=service, options=options)

    # Load Slickcharts page
    url = "https://www.slickcharts.com/sp500"
    driver.get(url)
    time.sleep(3)  # wait for page to fully load

    # Parse the table
    table = driver.find_element(By.TAG_NAME, "table")
    rows = table.find_elements(By.TAG_NAME, "tr")

    # Extract data
    data = []
    for row in rows[1:]:  # skip header
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) == 0:
            continue
        rank = cols[0].text
        company = cols[1].text
        symbol = cols[2].text
        weight = float(cols[3].text.strip("%")) / 100  # convert to decimal
        price = float(cols[4].text.replace(",", ""))
        chg = cols[5].text
        chg_pct = cols[6].text
        data.append([rank, company, symbol, weight, price, chg, chg_pct])
    # Convert to DataFrame
    df1 = pd.DataFrame(
        data,
        columns=["Rank", "Company", "Symbol", "Weight", "Price", "Change", "% Change"],
    )

    # Open S and P wiki page to get sector information
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Open the page
    driver.get(url)

    # Wait for the page to load (can increase this if necessary)
    driver.implicitly_wait(10)

    # Locate the table by its class name
    table = driver.find_element(By.CLASS_NAME, "wikitable")

    # Extract the table rows
    rows = table.find_elements(By.TAG_NAME, "tr")

    # Extract headers
    headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, "th")]

    # Extract table data
    data = []
    for row in rows[1:]:
        columns = row.find_elements(By.TAG_NAME, "td")
        data.append([col.text for col in columns])

    # Convert to a DataFrame
    df2 = pd.DataFrame(data, columns=headers)
    merged_df = pd.merge(df1, df2, on="Symbol", how="inner")
    merged_df["Symbol"] = merged_df["Symbol"].str.replace(".", "-", regex=False)
    driver.quit()
    return merged_df


@st.cache_data(show_spinner=True)
def download_stock_data(tickers, start_date, end_date):
    """
    This function will download data from
    yfinace api using given list of tickers
    start data and end date at a resolution of 1 day.
    It will return the 1 day percentage change in returns

    Args:
        tickers (list): ticker list
        start_date (string): representing the start date in 'YYYY-MM-DD' format
        end_date (string): representing the end date in 'YYYY-MM-DD' format

    Returns:
        dataframe: pandas DataFrame containing the stock data
    """

    tickers = tickers + ["^GSPC"]
    data = yf.download(tickers, start=start_date, end=end_date)
    data_close = data["Close"]
    # data_returns = data_close.pct_change.dropna()
    # cleaned_data = remove_outliers_3sigma(data_returns)
    # filtered_data = data_close[data_close.index.isin(cleaned_data.index)]

    return data_close  # this is not standarsized still, it also contain S and P index data (remove it before running PCA)
