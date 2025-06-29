import os
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime

SOURCE_INDEX_PATH = "data/fonte_index.json"

# === STEP 1: Download HTML ===
def get_clean_filename_from_url_path(path):
    return path.strip("/").replace("/", "_") or "index"

def download_html(url, output_dir="data/raw"):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Download error [{response.status_code}]: {url}")
        return None, None

    filename = get_clean_filename_from_url_path(urlparse(url).path) + ".html"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"HTML saved to: {output_path}")
    return filename, output_path

# === STEP 2: Manage source index ===
def load_source_index():
    if os.path.exists(SOURCE_INDEX_PATH):
        with open(SOURCE_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_source_index(index):
    os.makedirs(os.path.dirname(SOURCE_INDEX_PATH), exist_ok=True)
    with open(SOURCE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

def register_source_metadata(filename, url):
    print("\nEnter source metadata:")
    title = input("Title: ").strip()
    subject = input("Subject (e.g. storia, geografia): ").strip().lower()
    level = input("Level (elementari / medie / superiori / uni): ").strip().lower()

    index = load_source_index()
    source = {
        "id": filename.replace(".html", ""),
        "titolo": title,
        "materia": subject,
        "livello": level,
        "fonte": url,
        "formato": "html",
        "salvato_il": datetime.now().isoformat()
    }
    index.append(source)
    save_source_index(index)
    print("Source registered in fonte_index.json\n")

# === STEP 3: Extract clean text ===
def extract_text_from_html(html_path, output_path=None):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    if soup.find("div", {"id": "mw-content-text"}):
        content = soup.find("div", {"id": "mw-content-text"})
    elif soup.find("article"):
        content = soup.find("article")
    else:
        content = soup.body

    tags = content.find_all(["h1", "h2", "h3", "p", "li"])
    text = "\n\n".join([t.get_text(strip=True) for t in tags]).strip()

    if not output_path:
        output_path = html_path.replace("data/raw/", "data/cleaned/").replace(".html", ".txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Clean text saved to: {output_path}")
    return output_path

# === Main pipeline ===
if __name__ == "__main__":
    url = input("Enter the source URL: ").strip()
    
    filename, html_path = download_html(url)
    if filename and html_path:
        register_source_metadata(filename, url)
        extract_text_from_html(html_path)
