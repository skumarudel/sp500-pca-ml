from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_PCA(data, n_components, normalize):
    """Run PCA on the given data.
    This function performs Principal Component Analysis (PCA) on the provided dataset.
    It allows for normalization of the data and specifies the number of principal components to retain.

    Args:
        data (dataframe): pandas DataFrame containing the data to be analyzed.
        n_components (int): Number of principal components to retain.
        normalize (boolean): Whether to normalize the data before applying PCA.

    Returns:
        _type_: A tuple containing the PCA object and the transformed data.
    """
    if "^GSPC" in data.columns:
        data_cleaned = data.drop(columns=["^GSPC"])
    else:
        data_cleaned = data.copy()

    returns = data_cleaned.pct_change().dropna()
    if normalize:
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)
    else:
        returns_scaled = returns
    pca = PCA(n_components=n_components)
    returns_pca = pca.fit_transform(returns_scaled)
    return returns, pca, returns_pca
