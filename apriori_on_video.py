from ultralytics import YOLO
import cv2
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
model_basket = YOLO('models/basketv8.pt')
model_product = YOLO('models/shelfv8.pt')
product_classes = ['50 50 Biscuit', 'Biscafe', 'Bounce', 'Bourbon Dark fantasy', 'Bourbon', 'Bourn Vita Biscuit', 'Chocobakes', 'Coffee Joy', 'Creme', 'Dark Fantasy', 'Digestive', 'Elite', 'Ginger', 'Good Day', 'Happy Happy', 'Hide - Seek', 'Jim Jam', 'KrackJack', 'Malkist', 'Marie Gold', 'Marie Light', 'Milk Bikis', 'Milk Short Cake', 'Mom Magic', 'Monaco', 'Nice', 'Nutri Choice', 'Nutri Choice-Crackers-', 'Nutri Choice-Herbs-', 'Nutri Choice-Sugar Free-', 'Oreo', 'Parle G', 'Potazo', 'Sunfeast green', 'Super Millets', 'Supermilk', 'Tninz', 'Treat', 'Unibic', 'Unibic-box', 'allrounder']
data = pd.read_csv("random_transaction_data.csv")

cap = cv2.VideoCapture('sample_video/basketvideo_Trim.mp4')
basket_id = {}
product_id = {}
t_id = {}
processed_products = {}  # Create a dictionary to track processed products for each transaction ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color = (0, 255, 255)
    results = model_basket.track(frame, mode='track', conf=0.4)

    for result in results:
        if result.boxes.id is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()

        for box, id in zip(boxes, ids):
            bx1, by1, bx2, by2 = map(int, box)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(frame, "Basket", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            results_product = model_product.track(frame, mode='track', conf=0.1)
            products_in_basket = []
            for result_product in results_product:
                if result_product.boxes.id is None:
                    continue

                boxes_product = result_product.boxes.xyxy.cpu().numpy()
                classes_product = result_product.boxes.cls.cpu().numpy()
                ids_product = result_product.boxes.id.cpu().numpy()
                print("ids_product",ids_product)
                print("classes_product", classes_product)

                for box_p, label_p, id_p in zip(boxes_product, classes_product, ids_product):
                    print("id_p:",id_p)
                    px1, py1, px2, py2 = map(int, box_p)
                    if px1 >= bx1 and px2 <= bx2 and py1 >= by1 and py2 <= by2:
                        product_name = product_classes[int(label_p)]
                        print(product_name)
                        if id in basket_id:
                            basket_id[id][id_p] = product_name
                            products_in_basket.append(product_name)  # Add product name to the list

                        else:
                            basket_id[id] = {id_p: product_name}
                            max_transaction_id = data["Transaction-ID"].max()+1
                            t_id[id] = max_transaction_id
                            next_transaction_id = max_transaction_id
                            new_row = pd.DataFrame({'Transaction-ID': [next_transaction_id]})

                            # Add columns for all other products and set them to zero
                            product_columns = data.columns[1:]  # Exclude the 'Transaction-ID' column
                            for column in product_columns:
                                new_row[column] = 0

                            # Append the new row to the existing DataFrame
                            data = data.append(new_row, ignore_index=True)

                           
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,0), 2)
                        cv2.putText(frame, product_name+" "+str(id_p), (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            if products_in_basket:
                products_text = ", ".join(products_in_basket)
                cv2.putText(frame, "Products: " + products_text, (bx1, by1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(basket_id)
    for transaction_id, products in basket_id.items():
        transaction_id = t_id[transaction_id]
        if transaction_id in data["Transaction-ID"].values:
            # Get the row index where Transaction-ID matches
            row_index = data[data["Transaction-ID"] == transaction_id].index[0]
            
            # Check if this transaction ID has been processed in the current frame
            if transaction_id not in processed_products:
                processed_products[transaction_id] = set()  # Initialize the set for this transaction ID

            # Loop through the products in the dictionary and update the corresponding columns
            for product_id, product_name in products.items():
                if product_name not in processed_products[transaction_id]:
                    # Find the corresponding product column in the transaction data based on product_name
                    print("product id:", product_id)
                    print("product name",product_name)
                    column_name = data.columns[data.columns.str.contains(product_name)].item()
                    print("column name", column_name)
                    # Update the corresponding cell in the transaction data
                    data.at[row_index, column_name] += 1
                    processed_products[transaction_id].add(product_name)

    data.to_csv("random_transaction_data.csv", index=False)
    basket = data.drop(columns=["Transaction-ID"])  # Exclude the Transaction-ID column
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Apply the Apriori algorithm to find frequent itemsets with a minimum support threshold (e.g., 0.2)
    frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

    # Generate association rules with a minimum confidence threshold (e.g., 0.5)
    association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    # Sort the association rules by lift (a measure of how strongly two items are associated)
    association_rules_df = association_rules_df.sort_values(by="lift", ascending=False)

    # Print the best product recommendations and closely associated products

    best_product_recommendation = association_rules_df.head(1).iloc[0]

    # Extract antecedents and consequents
    antecedents = best_product_recommendation['antecedents']
    consequents = best_product_recommendation['consequents']

    # Extract antecedent and consequent product names
    antecedent_products = [str(product) for product in antecedents]
    consequent_products = [str(product) for product in consequents]

    # Create the recommendation description
    recommendation_description = f"That people who bought {', '.join(antecedent_products)} are likely to also buy {', '.join(consequent_products)} - Apriori Algorithm."
    cv2.putText(frame, recommendation_description, (20, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0, 0), 2)
    # Print the recommendation description
    print(recommendation_description)
    closely_associated_products = association_rules_df.head(5)

    print("\nClosely Associated Products:")
    print(closely_associated_products)
    cv2.imshow("hello", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
