import time
import pandas as pd
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def read_transactions(path):
	"""Read the CSV where each row is a comma-separated transaction."""
	transactions = []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			parts = [p.strip() for p in line.strip().split(',') if p.strip()]
			if parts:
				transactions.append(parts)
	return transactions


def eclat_bruteforce(transactions, min_support=0.05):
	"""Simple Eclat-style frequent itemset discovery by counting combinations.
	This is a brute-force implementation suitable for small item counts.
	Returns a DataFrame with columns ['support', 'itemsets'].
	"""
	n = len(transactions)
	# unique items
	items = sorted({item for t in transactions for item in t})
	results = []
	max_k = len(items)
	for k in range(1, max_k + 1):
		any_found = False
		for combo in combinations(items, k):
			count = 0
			combo_set = set(combo)
			for t in transactions:
				if combo_set.issubset(t):
					count += 1
			support = count / n
			if support >= min_support:
				any_found = True
				results.append({'support': support, 'itemsets': frozenset(combo)})
		if not any_found:
			# no frequent itemsets of this size -> larger sizes will not be frequent
			break
	if results:
		df = pd.DataFrame(results).sort_values(by=['support', 'itemsets'], ascending=[False, True])
	else:
		df = pd.DataFrame(columns=['support', 'itemsets'])
	return df


def main():
	data_path = 'Breakfast_Transactions.csv'
	print('Reading transactions from', data_path)
	transactions = read_transactions(data_path)
	print(f'Number of transactions: {len(transactions)}')

	# One-hot encode
	te = TransactionEncoder()
	te_ary = te.fit(transactions).transform(transactions)
	df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

	min_support = 0.05
	min_confidence = 0.6

	# Apriori
	t0 = time.perf_counter()
	fi_apriori = apriori(df_onehot, min_support=min_support, use_colnames=True)
	t1 = time.perf_counter()
	apriori_time = t1 - t0
	print('\nApriori found', len(fi_apriori), 'itemsets in', f'{apriori_time:.3f}s')
	fi_apriori.to_csv('frequent_itemsets_apriori.csv', index=False)

	# FP-Growth
	t0 = time.perf_counter()
	fi_fpgrowth = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
	t1 = time.perf_counter()
	fpgrowth_time = t1 - t0
	print('FPGrowth found', len(fi_fpgrowth), 'itemsets in', f'{fpgrowth_time:.3f}s')
	fi_fpgrowth.to_csv('frequent_itemsets_fpgrowth.csv', index=False)

	# Eclat (brute-force)
	t0 = time.perf_counter()
	fi_eclat = eclat_bruteforce(transactions, min_support=min_support)
	t1 = time.perf_counter()
	eclat_time = t1 - t0
	print('Eclat (bruteforce) found', len(fi_eclat), 'itemsets in', f'{eclat_time:.3f}s')
	fi_eclat.to_csv('frequent_itemsets_eclat.csv', index=False)

	# Generate association rules for apriori results
	rules_apriori = association_rules(fi_apriori, metric='confidence', min_threshold=min_confidence)
	rules_fpgrowth = association_rules(fi_fpgrowth, metric='confidence', min_threshold=min_confidence)
	# For eclat, association_rules expects a DataFrame with itemsets in 'itemsets' column; our df matches that
	rules_eclat = association_rules(fi_eclat.rename(columns={'itemsets': 'itemsets', 'support': 'support'}),
									metric='confidence', min_threshold=min_confidence) if not fi_eclat.empty else pd.DataFrame()

	print('\nRules found:')
	print('Apriori rules:', len(rules_apriori))
	print('FPGrowth rules:', len(rules_fpgrowth))
	print('Eclat rules:', len(rules_eclat))

	# Save rules
	rules_apriori.to_csv('rules_apriori.csv', index=False)
	rules_fpgrowth.to_csv('rules_fpgrowth.csv', index=False)
	rules_eclat.to_csv('rules_eclat.csv', index=False)

	# Summary
	print('\nSummary:')
	print(f'Apriori: {len(fi_apriori)} itemsets, {len(rules_apriori)} rules, time={apriori_time:.3f}s')
	print(f'FPGrowth: {len(fi_fpgrowth)} itemsets, {len(rules_fpgrowth)} rules, time={fpgrowth_time:.3f}s')
	print(f'Eclat: {len(fi_eclat)} itemsets, {len(rules_eclat)} rules, time={eclat_time:.3f}s')


if __name__ == '__main__':
	main()
