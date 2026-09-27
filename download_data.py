"""
Download the UCI Heart Disease dataset from a public source.
This is the commonly used Cleveland heart disease dataset.
"""
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

DEST = os.path.join(DATA_DIR, "heart.csv")

# The standard Kaggle/UCI heart disease dataset (Cleveland)
# This URL hosts the commonly used version with 303 samples and 14 columns
URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"

# Alternative URLs if the above fails
ALT_URLS = [
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
]

def download():
    if os.path.exists(DEST):
        print(f"Dataset already exists at: {DEST}")
        return
    
    urls_to_try = [URL] + ALT_URLS
    for url in urls_to_try:
        try:
            print(f"Downloading from: {url}")
            urllib.request.urlretrieve(url, DEST)
            print(f"Dataset saved to: {DEST}")
            
            # Verify the file is not empty
            size = os.path.getsize(DEST)
            if size < 100:
                os.remove(DEST)
                print(f"  File too small ({size} bytes), trying next URL...")
                continue
            
            print(f"  File size: {size:,} bytes")
            return
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print("\nAll download URLs failed.")
    print("Please manually download the heart disease dataset and place it at:")
    print(f"  {DEST}")
    print("\nRecommended source:")
    print("  https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")

if __name__ == "__main__":
    download()
