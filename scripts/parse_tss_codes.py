# scripts/parse_tss_codes.py
import argparse
from bs4 import BeautifulSoup
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Parse TCGA TSS code table from GDC HTML page")
    parser.add_argument("--input_html", default="data/reference/tcga_tss_codes.html")
    parser.add_argument("--output_csv", default="data/reference/tcga_tss_codes.csv")
    args = parser.parse_args()

    with open(args.input_html, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    target_table = None
    for t in soup.find_all("table"):
        thead = t.find("thead")
        if thead and "TSS Code" in thead.get_text():
            target_table = t
            break


    if target_table is None:
        raise RuntimeError("Could not find a table with a 'TSS Code' header on this page.")

    rows = []
    for tr in target_table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == 4:
            rows.append(cells)

    df = pd.DataFrame(rows, columns=["tss_code", "source_site", "study_name", "bcr"])
    df.to_csv(args.output_csv, index=False)
    print(f"Parsed {len(df)} TSS code rows -> {args.output_csv}")
    print(df["study_name"].value_counts().head(10))
    print(df["tss_code"].head(15).tolist())

if __name__ == "__main__":
    main()
