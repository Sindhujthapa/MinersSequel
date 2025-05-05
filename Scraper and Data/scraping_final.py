import pandas as pd
import requests
import csv
import urllib.parse
import time
import random

# Unique game IDs from site
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
        "language": "english",
        "day_range": 365,
        "num_per_page": 100,
        "review_type": "all",
        "purchase_type": "all",
        "filter": "all"
    }
    
    cursor = "*"
    consecutive_empty_pages = 0
    max_consecutive_empty = 3  # After this many empty pages, try new parameters
    
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
                        cursor = "*"  # Reset cursor
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
                
                # Only add if we have actual review text
                if review_text.strip():
                    all_reviews.append([rec_id, language, voted_up, app_id, review_text])
            
            # If no new reviews were found in this batch
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

def save_reviews_to_csv(all_data, filename="steam_reviews.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["recommendationid", "language", "voted_up", "app_id", "review"])
        writer.writerows(all_data)
    print(f"Saved {len(all_data)} reviews to {filename}")

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
            save_reviews_to_csv(combined_data, f"steam_reviews_interim_{i+1}.csv")
    

    unique_ids = set()
    final_data = []
    
    for review in combined_data:
        rec_id = review[0]  
        if rec_id not in unique_ids:
            unique_ids.add(rec_id)
            final_data.append(review)
    
    print(f"\nFinal check: {len(final_data)} unique reviews out of {len(combined_data)} collected")
    save_reviews_to_csv(final_data, "steam_reviews_unique.csv")

    # Clean csv file
    df = pd.read_csv("steam_reviews_unique.csv")
    df["review"] = df["review"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip().str.replace("\u00A0", " ", regex = False)
    df_unique = df.drop_duplicates(subset = "review")
    df_unique.to_csv("steam_reviews_unique.csv", index=False, encoding="utf-8-sig")
    
if __name__ == "__main__":
    main()
