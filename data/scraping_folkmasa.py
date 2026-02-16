import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from urllib.parse import urljoin


INDEX_URL     = "https://folkmasa.org/yashpeh/mb_yash.php"
BASE_URL      = "https://folkmasa.org/yashpeh/"
HEADERS       = {
    "User-Agent": "FolkTalesResearchBot - academic/non-commercial use only"
}
DELAY         = 7           
OUTPUT_FILE   = "yashpeh_folktales_with_book_tradition.csv"

def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_index():
    print("Parsing index page...")
    soup = get_soup(INDEX_URL)
    
    stories = []
    
    tables = soup.find_all("table", {"width": ["90%", "85%"], "align": "center"})
    if not tables:
        tables = soup.find_all("table")
    
    for table in tables:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            
            left_td, middle_td, right_td = tds[:3]
            
            link = right_td.find("a", href=re.compile(r'mb_yashp\.php\?mishtane=\d+'))
            if not link:
                continue
                
            href = link["href"]
            mishtane_match = re.search(r'mishtane=(\d+)', href)
            if not mishtane_match:
                continue
            story_id = mishtane_match.group(1)
            story_url = urljoin(BASE_URL, href)
            
            title_tag = middle_td.find(["span", "p", "font", "div"])
            title = title_tag.get_text(strip=True) if title_tag else middle_td.get_text(strip=True)
            
            if not title or len(title) < 1 or "next" in title.lower():
                continue
            
            stories.append({
                "title": title,
                "url": story_url,
                "id": story_id
            })
    
    print(f"Found {len(stories)} story links.")
    return stories

def extract_story_metadata_and_text(url):
    try:
        soup = get_soup(url)
        
        book_name = ""
        tradition = ""
        text_parts = []
        

        tables = soup.find_all("table", align="center")
        main_table = None
        
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) >= 5:
                row2_text = rows[1].get_text(" ", strip=True).lower()
                row3_text = rows[2].get_text(" ", strip=True).lower()
                if "book name" in row2_text and "tradition" in row3_text:
                    main_table = table
                    break
        
        if main_table:
            rows = main_table.find_all("tr")
            
            # Row 1 
            title_cell = rows[0].find("td")
            if title_cell:
                title = title_cell.get_text(strip=True)
            
            # Row 2 
            book_cell = rows[1].find("td")
            if book_cell:
                txt = book_cell.get_text(":", strip=True)
                if ":" in txt:
                    book_name = txt.split(":", 1)[1].strip()
                else:
                    book_name = txt.strip()
            
            # Row 3, tradition
            trad_cell = rows[2].find("td")
            if trad_cell:
                txt = trad_cell.get_text(":", strip=True)
                if ":" in txt:
                    tradition = txt.split(":", 1)[1].strip()
                else:
                    tradition = txt.strip()
            
            # Row 5, text 
            if len(rows) >= 5:
                content_td = rows[4].find("td")
                if content_td:
                    raw_text = content_td.get_text(separator="\n", strip=True)
                    lines = [re.sub(r'\s+', ' ', line).strip() for line in raw_text.splitlines() if line.strip()]
                    text_parts = [line for line in lines if len(line) > 1]
        

        if not text_parts:
            print(f"  Warning: No 5-row table {url}")
            all_text = soup.body.get_text(separator="\n", strip=True)
            lines = [re.sub(r'\s+', ' ', l).strip() for l in all_text.splitlines() if l.strip()]
            
            in_content = False
            for line in lines:
                lower = line.lower()
                if "book name:" in lower:
                    book_name = line.split(":", 1)[1].strip() if ":" in line else ""
                elif "tradition:" in lower:
                    tradition = line.split(":", 1)[1].strip() if ":" in line else ""
                elif len(line) > 1 and "." in line and not any(x in lower for x in ["comments:", "abstract:", "to next story"]):
                    in_content = True
                if in_content:
                    text_parts.append(line)
        
        full_text = "\n\n".join(text_parts).strip() if text_parts else "[TEXT NOT FOUND]"
        
        return {
            "book_name": book_name,
            "tradition": tradition,
            "text": full_text
        }
    
    except Exception as e:
        print(f"Error on {url}: {e}")
        return {
            "book_name": "",
            "tradition": "",
            "text": f"[ERROR: {str(e)}]"
        }

def main():
    stories = parse_index()
    
    if not stories:
        print("No stories detected.")
        return
    

    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["title", "book_name", "tradition", "url", "story_id", "text"])
        
        for i, s in enumerate(stories, 1):
            print(f"[{i:3d}/{len(stories):3d}] {s['title'][:70]}...")
            data = extract_story_metadata_and_text(s["url"])
            
            writer.writerow([
                s["title"],
                data["book_name"],
                data["tradition"],
                s["url"],
                s["id"],
                data["text"]
            ])
            
            time.sleep(DELAY)
    
    print(f"\nFinished. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()