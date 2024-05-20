import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load your transaction dataset
data = pd.read_csv("transaction_data1.csv")

# Convert the dataset to binary format where 0 means the product was not bought and 1 means it was bought
basket = data.drop(columns=["Transaction-ID"])  # Exclude the Transaction-ID column
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Apply the Apriori algorithm to find frequent itemsets with a minimum support threshold (e.g., 0.2)
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

# Generate association rules with a minimum confidence threshold (e.g., 0.5)
association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Sort the association rules by lift (a measure of how strongly two items are associated)
association_rules_df = association_rules_df.sort_values(by="lift", ascending=False)

# Print the best product recommendations and closely associated products
best_product_recommendations = association_rules_df.head(1)
closely_associated_products = association_rules_df.head(5)

print("Best Product Recommendations:")
print(best_product_recommendations)

print("\nClosely Associated Products:")
print(closely_associated_products)