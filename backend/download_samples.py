import urllib.request
from pathlib import Path

def download_sample_data():
    # Define destination path
    dest_dir = Path("data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = dest_dir / "brain_tumors_ucni.pdf"
    
    # URL to the reference PDF used in the developer project
    url = "https://raw.githubusercontent.com/souvikmajumder26/Multi-Agent-Medical-Assistant/main/data/raw/brain_tumors_ucni.pdf"
    
    print(f"Downloading sample medical guideline PDF to: {file_path}...")
    try:
        urllib.request.urlretrieve(url, file_path)
        print("Download completed successfully!")
        print("\nYou can now:")
        print("1. Upload this file using the React Dashboard's 'Literature Ingest' panel.")
        print("2. Or run: python ingest.py --file data/raw/brain_tumors_ucni.pdf")
    except Exception as e:
        print(f"Error downloading file: {e}")

if __name__ == "__main__":
    download_sample_data()
