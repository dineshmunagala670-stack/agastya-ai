# dataset/fetch_large_data.py
import urllib.request
import os

# A list of massive, clean plain-text literary masterpieces (Zero code, zero markup)
BOOK_URLS = {
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "pride_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "dracula": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "time_machine": "https://www.gutenberg.org/files/35/35-0.txt",
    "moby_dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
}

OUTPUT_PATH = "dataset/large_input.txt"

# Clean out the old messy wikipedia files if they exist
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)
if os.path.exists("dataset/enwik8.zip"):
    os.remove("dataset/enwik8.zip")

print("--- Building Project Agastya Pristine Dataset ---")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
    for book_name, url in BOOK_URLS.items():
        print(f"Downloading and appending {book_name}...")
        try:
            with urllib.request.urlopen(url) as response:
                book_text = response.read().decode('utf-8', errors='ignore')
                # Write to database file
                outfile.write(book_text + "\n\n")
        except Exception as e:
            print(f"Skipped {book_name} due to an adjustment error: {e}")

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"\nSuccess! New high-quality text dataset compiled.")
print(f"Final Size: {file_size_mb:.2f} MB of pure, clean human grammar.")