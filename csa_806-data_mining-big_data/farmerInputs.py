import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time

# Create synthetic agricultural transactions
def create_dataset():
    data = []
    items = ['Maize', 'Wheat', 'Soybeans', 'DAP', 'Urea', 'CAN', 'Herbicide', 'Pesticide', 'Fungicide']
    
    for _ in range(1000):
        transaction = []
        # Bias: Maize farmers almost always buy DAP and Herbicide
        if np.random.rand() < 0.4:
            transaction.extend(['Maize', 'DAP', 'Herbicide'])
        # Bias: Wheat farmers often buy Urea and Fungicide
        elif np.random.rand() < 0.3:
            transaction.extend(['Wheat', 'Urea', 'Fungicide'])
        # Add some random items
        num_extras = np.random.randint(1, 3)
        transaction.extend(np.random.choice(items, num_extras).tolist())
        data.append(list(set(transaction))) # unique items per transaction
    return data

dataset = create_dataset()

# preprocess the dataset
# one-hot encode the dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
#print dataset shape
print("Dataset shape:", df.shape)

# Algorithm 1: Apriori
start_time = time.time()
frequent_itemsets_apriori = apriori(df, min_support=0.1, use_colnames=True)
apriori_time = time.time() - start_time

# Algorithm 2: FP-Growth
start_time = time.time()
frequent_itemsets_fp = fpgrowth(df, min_support=0.1, use_colnames=True)
fpgrowth_time = time.time() - start_time

print(f"Apriori Time: {apriori_time:.4f}s")
print(f"FP-Growth Time: {fpgrowth_time:.4f}s")

# Generate association rules
rules = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1.2)
# Sorting by confidence to find the most 'certain' bundles
top_rules = rules.sort_values('confidence', ascending=False).head(10)

# Displaying key columns
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])