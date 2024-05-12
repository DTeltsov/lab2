from itertools import combinations


def load_data():
    return [
        [('a', 'b', 'c', 'd'), ('b', 'c', 'd'), ('a', 'e', 'f', 'g', 'h'), ('b', 'c', 'd', 'e', 'g', 'j'),
         ('b', 'c', 'd', 'e', 'f'),
         ('a', 'f', 'g'), ('a', 'i', 'j'), ('a', 'b', 'e', 'h'), ('f', 'g', 'h', 'i', 'j'), ('e', 'f', 'h')],
        [('a', 'b', 'c', 'f'), ('b', 'c', 'f'), ('b', 'd', 'e', 'g', 'h'), ('b', 'c', 'e', 'f', 'g', 'j'),
         ('b', 'c', 'd', 'e', 'f'),
         ('a', 'd', 'g'), ('a', 'i', 'j'), ('a', 'b', 'e', 'h'), ('d', 'g', 'h', 'i', 'j'), ('d', 'e', 'h')],
        [('a', 'b', 'c', 'd', 'e', 'f'), ('b', 'c', 'd'), ('b', 'e', 'g', 'h'), ('b', 'c', 'g', 'j'),
         ('b', 'c', 'd', 'e', 'f'),
         ('a', 'e', 'f'), ('a', 'i', 'j'), ('a', 'b', 'c', 'e', 'h'), ('e', 'f', 'h', 'i', 'j'), ('b', 'f', 'h')]
    ]


def apriori(transactions, min_support, min_confidence):
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions

    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1

    current_itemsets = {frozenset([item]): count for item, count in item_counts.items() if count >= min_support_count}
    frequent_itemsets = dict(current_itemsets)

    k = 2
    while current_itemsets:
        new_candidates = set()
        itemsets = list(current_itemsets.keys())
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                union_set = itemsets[i].union(itemsets[j])
                if len(union_set) == k:
                    new_candidates.add(union_set)

        candidate_counts = {candidate: 0 for candidate in new_candidates}
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in new_candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1

        current_itemsets = {itemset: count for itemset, count in candidate_counts.items() if count >= min_support_count}
        frequent_itemsets.update(current_itemsets)
        k += 1

    rules = []
    for itemset, support in frequent_itemsets.items():
        for conseq_length in range(1, len(itemset)):
            for consequence in combinations(itemset, conseq_length):
                premise = itemset - set(consequence)
                if premise:
                    premise_support = frequent_itemsets[frozenset(premise)]
                    confidence = support / premise_support
                    if confidence >= min_confidence:
                        rules.append((premise, consequence, confidence, support / num_transactions))
    return rules


def print_rules(rules):
    for rule in rules:
        premise, consequence, confidence, support = rule
        premise_str = ', '.join(premise)
        consequence_str = ', '.join(consequence)
        print(f"Правило: {{{premise_str}}} -> {{{consequence_str}}}, Достовірність: {confidence * 100:.2f}%, Підтримка: {support * 100:.2f}%")



datasets = load_data()
all_rules = []
min_support = 0.4
min_confidence = 0.75

for i, dataset in enumerate(datasets, start=1):
    print(f"Множина транзакцій {i}:")
    rules = apriori(dataset, min_support, min_confidence)
    print_rules(rules)
    print("\n")
