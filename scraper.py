import requests
import csv
import urllib.parse
import time

# Editable list of game app IDs
APP_IDS = ['949230', '424370','952070','360430']  # Add more appids as needed

# API URL parameters
BASE_URL = "https://store.steampowered.com/appreviews/{appid}"
PARAMS = {
    "json": 1,
    "filter": "all",
    "language": "english",
    "day_range": 365,
    "num_per_page": 100,
}

def fetch_reviews_for_app(app_id, max_reviews=500):
    all_reviews = []
    cursor = "*"
    num_pages = max_reviews // 100

    for _ in range(num_pages):
        params = PARAMS.copy()
        params["cursor"] = cursor
        url = BASE_URL.format(appid=app_id)
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Failed to fetch reviews for app {app_id}, status code: {response.status_code}")
            break

        data = response.json()
        reviews = data.get("reviews", [])
        for review in reviews:
            rec_id = review.get("recommendationid")
            language = review.get("language")
            voted_up = 1 if review.get("voted_up") else 0
            review_text = review.get("review", "").replace('\n', ' ').strip()
            all_reviews.append([rec_id, language, voted_up, app_id, review_text])

        # Update cursor and URL encode it
        raw_cursor = data.get("cursor")
        if not raw_cursor:
            print(f"No more reviews available for app {app_id}.")
            break
        cursor = urllib.parse.quote(raw_cursor)
        time.sleep(1)

    return all_reviews

def save_reviews_to_csv(all_data, filename="steam_reviews.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["recommendationid", "language", "voted_up", "app_id", "review"])
        writer.writerows(all_data)
    print(f"Saved {len(all_data)} reviews to {filename}")

def main():
    combined_data = []
    for app_id in APP_IDS:
        print(f"Fetching reviews for app {app_id}...")
        app_reviews = fetch_reviews_for_app(app_id)
        combined_data.extend(app_reviews)

    save_reviews_to_csv(combined_data)

if __name__ == "__main__":
    main()
