"""
Use PCA to analyze stock market data and identify patterns.

Tasks:

Obtain and preprocess a dataset of stock prices.
Implement PCA to reduce dimensionality and visualize the principal components.
Interpret the principal components and their impact on the dataset
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Obtain and preprocess a dataset of stock prices
# For demonstration, we'll use Yahoo Finance data via pandas_datareader if available,
# otherwise, we'll simulate some stock data.

def get_stock_data(symbols, start_date, end_date):
    try:
        import pandas_datareader.data as web
        df = pd.DataFrame()
        for symbol in symbols:
            data = web.DataReader(symbol, 'yahoo', start_date, end_date)['Adj Close']
            df[symbol] = data
        return df
    except Exception as e:
        print("Could not fetch real stock data, simulating random data instead.")
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        data = np.cumsum(np.random.randn(len(dates), len(symbols)), axis=0) + 100
        df = pd.DataFrame(data, index=dates, columns=symbols)
        return df

# Define stock symbols and date range
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'UNH']
start_date = '2022-01-01'
end_date = '2022-12-31'

# Get the stock data
stock_data = get_stock_data(symbols, start_date, end_date)

# Drop rows with missing values (if any)
stock_data = stock_data.dropna()

# 2. Preprocess: Calculate daily returns and standardize
returns = stock_data.pct_change().dropna()
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# 3. Implement PCA
pca = PCA()
pca.fit(returns_scaled)
explained_variance = pca.explained_variance_ratio_

# Project the data onto the first two principal components
pca_2 = PCA(n_components=2)
principal_components = pca_2.fit_transform(returns_scaled)

# 4. Visualize the principal components
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
plt.title('Stock Returns Projected onto First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Plot explained variance ratio
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by Principal Components')
plt.show()

# 5. Interpret the principal components
print("Explained variance ratio for each principal component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var*100:.2f}%")

# Show the loadings (contribution of each stock to the first two PCs)
loadings = pd.DataFrame(
    pca.components_[:2].T,
    columns=['PC1', 'PC2'],
    index=symbols
)
print("\nPrincipal Component Loadings (first two PCs):")
print(loadings)

print("\nInterpretation:")
print("- The explained variance ratio shows how much of the total variance in the stock returns is captured by each principal component.")
print("- The loadings indicate which stocks contribute most to each principal component. Large positive or negative values mean a strong influence.")
print("- By examining the loadings, you can identify groups of stocks that move together (e.g., tech stocks) or stocks that behave differently from the rest.")
