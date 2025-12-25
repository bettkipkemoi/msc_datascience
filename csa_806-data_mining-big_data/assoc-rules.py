from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
# Sample dataset
data = {'Tomatoes': [1, 1, 0, 1], 'Onions': [1, 0, 1, 1], 'Garlic': [0, 1, 1,
1]}
df = pd.DataFrame(data)
# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets)
# Generate association rules
145
rules = association_rules(frequent_itemsets, metric="confidence",
min_threshold=0.7)
print(rules)