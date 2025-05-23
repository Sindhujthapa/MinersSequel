import pandas as pd
import requests
import csv
import urllib.parse
import time
import random

# Unique Game IDs from sites
APP_IDS = ['3164500', '2379780', '3146520', '1089350', '954850', '2246340', '1128000', '105600', '620', '646910', '2446550', '360430', '952070', '424370', '949230', '1778820', '1811260', '1938090', '1097150']

BASE_URL = "https://store.steampowered.com/appreviews/{appid}"

def fetch_reviews_for_app(app_id, target_reviews=500):
    all_reviews = []
    seen_ids = set()  
    
    filter_options = ["all", "recent", "updated"]
    review_type_options = ["all", "positive", "negative"]
    purchase_type_options = ["all", "steam", "non_steam_purchase"]
    
    params = {
        "json": 1,
        "language": "spanish",
        "day_range": 365,
        "num_per_page": 100,
        "review_type": "all",
        "purchase_type": "all",
        "filter": "all"
    }
    
    cursor = "*"
    consecutive_empty_pages = 0
    max_consecutive_empty = 3  
    
    # Keep track of parameter combinations we've tried
    tried_combinations = set()
    
    while len(all_reviews) < target_reviews:
        current_params = params.copy()
        current_params["cursor"] = cursor
        param_key = f"{current_params['filter']}_{current_params['review_type']}_{current_params['purchase_type']}"
        
        if consecutive_empty_pages >= max_consecutive_empty:
            if len(tried_combinations) < len(filter_options) * len(review_type_options) * len(purchase_type_options):
                while True:
                    new_filter = random.choice(filter_options)
                    new_review_type = random.choice(review_type_options)
                    new_purchase_type = random.choice(purchase_type_options)
                    new_key = f"{new_filter}_{new_review_type}_{new_purchase_type}"
                    
                    if new_key not in tried_combinations:
                        params["filter"] = new_filter
                        params["review_type"] = new_review_type
                        params["purchase_type"] = new_purchase_type
                        tried_combinations.add(new_key)
                        print(f"Switching to new parameters: filter={new_filter}, review_type={new_review_type}, purchase_type={new_purchase_type}")
                        cursor = "*"  
                        consecutive_empty_pages = 0
                        break
            else:
                print(f"Tried all parameter combinations for app {app_id}. Collected {len(all_reviews)} reviews.")
                break
        
        url = BASE_URL.format(appid=app_id)
        
        try:
            response = requests.get(url, params=current_params)
            response.raise_for_status()
            
            data = response.json()
            reviews = data.get("reviews", [])
            
            new_reviews_count = 0
            
            for review in reviews:
                rec_id = review.get("recommendationid")
                
                # Skip if we've already seen this review
                if rec_id in seen_ids:
                    continue
                    
                seen_ids.add(rec_id)
                new_reviews_count += 1
                
                language = review.get("language")
                voted_up = 1 if review.get("voted_up") else 0
                review_text = review.get("review", "").replace('\n', ' ').strip()
                
                if review_text.strip():
                    all_reviews.append([rec_id, language, voted_up, app_id, review_text])
            
            if new_reviews_count == 0:
                consecutive_empty_pages += 1
                print(f"No new reviews found for app {app_id} with current parameters. Empty pages: {consecutive_empty_pages}")
            else:
                consecutive_empty_pages = 0  
                print(f"Found {new_reviews_count} new reviews for app {app_id}. Total: {len(all_reviews)}")
                
            raw_cursor = data.get("cursor")
            if not raw_cursor:
                print(f"No more pages available with current parameters for app {app_id}.")
                consecutive_empty_pages = max_consecutive_empty  
                continue
                
            cursor = urllib.parse.quote(raw_cursor)
            
            time.sleep(1.5)
            
        except Exception as e:
            print(f"Error fetching reviews for app {app_id}: {e}")
            time.sleep(5)  
            consecutive_empty_pages += 1
        
        if len(tried_combinations) > 10 and len(all_reviews) < target_reviews * 0.5:
            print(f"Unable to collect target number of reviews for app {app_id}. Stopping at {len(all_reviews)}.")
            break
    
    print(f"Completed fetching for app {app_id}. Total unique reviews: {len(all_reviews)}")
    return all_reviews

def save_reviews_to_csv(all_data, filename="steam_reviews_esp.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["recommendationid", "language", "voted_up", "app_id", "review"])
        writer.writerows(all_data)
    print(f"Saved {len(all_data)} reviews to {filename}")

def create_balanced_dataset(df):
    """
    Creates a balanced dataset with equal numbers of positive and negative reviews.
    """
    positive_reviews = df[df['voted_up'] == 1]
    negative_reviews = df[df['voted_up'] == 0]
    
    num_negative = len(negative_reviews)
    print(f"Found {len(positive_reviews)} positive reviews and {num_negative} negative reviews")
    
    if len(positive_reviews) > num_negative:
        positive_sample = positive_reviews.sample(n=num_negative, random_state=42)
        print(f"Sampling {num_negative} positive reviews to match negative reviews")
    else:
        positive_sample = positive_reviews
        print("Already balanced or more negative than positive reviews")
    
    balanced_df = pd.concat([positive_sample, negative_reviews])
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final balanced dataset size: {len(balanced_df)} reviews")
    print(f"Positive reviews: {len(balanced_df[balanced_df['voted_up'] == 1])}")
    print(f"Negative reviews: {len(balanced_df[balanced_df['voted_up'] == 0])}")
    
    return balanced_df

def main():
    combined_data = []
    total_unique_reviews = 0
    
    for i, app_id in enumerate(APP_IDS):
        print(f"\nProcessing game {i+1}/{len(APP_IDS)}: App ID {app_id}")
        app_reviews = fetch_reviews_for_app(app_id, target_reviews=500)
        combined_data.extend(app_reviews)
        total_unique_reviews += len(app_reviews)
        print(f"Progress: {len(combined_data)} total unique reviews collected")
        
        if (i + 1) % 5 == 0 or i == len(APP_IDS) - 1:
            save_reviews_to_csv(combined_data, f"steam_reviews_esp_interim_{i+1}.csv")
    
    
    unique_ids = set()
    final_data = []
    
    for review in combined_data:
        rec_id = review[0]  
        if rec_id not in unique_ids:
            unique_ids.add(rec_id)
            final_data.append(review)
    
    print(f"\nFinal check: {len(final_data)} unique reviews out of {len(combined_data)} collected")
    save_reviews_to_csv(final_data, "steam_reviews_unique_esp.csv")
    
    df = pd.read_csv("steam_reviews_unique_esp.csv")
    df["review"] = df["review"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip().str.replace("\u00A0", " ", regex=False)
    
    balanced_df = create_balanced_dataset(df)
    
    balanced_df.to_csv("steam_reviews_balanced_esp.csv", index=False, encoding="utf-8-sig")
    print("Balanced dataset saved to steam_reviews_balanced_esp.csv")
    
    print("\nClass Distribution in Balanced Dataset:")
    print(balanced_df['voted_up'].value_counts())
    print("\nPercentage:")
    print(balanced_df['voted_up'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    main()
