#!/usr/bin/env python3
import csv
import re
import time
import html
import urllib.parse
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
import pandas as pd

YEAR = 2021
HL = "en-GB"
GL = "GB"
CEID = "GB:en"
SLEEP_SECONDS = 0.6
OUTPUT_CSV = "hsbc_google_news_rss_2021.csv"

# Query variations to broaden coverage while staying stock/market focused.
QUERIES = [
    '"HSBC Holdings" stock',
    '"HSBC Holdings" shares',
    '"HSBC Holdings" London Stock Exchange',
    'HSBA LSE HSBC',
    'HSBC dividend',
    'HSBC earnings',
    'HSBC buyback',
    'HSBC results bank',
    'HSBC restructuring bank',
    'HSBC Evergrande',
]


def month_windows(year: int):
    for month in range(1, 13):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def build_rss_url(query: str, start_date: str, end_date: str) -> str:
    q = f'{query} after:{start_date} before:{end_date}'
    encoded = urllib.parse.quote_plus(q)
    return f'https://news.google.com/rss/search?q={encoded}&hl={HL}&gl={GL}&ceid={urllib.parse.quote_plus(CEID)}'


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_published(entry) -> str:
    for key in ("published", "updated"):
        value = entry.get(key)
        if value:
            try:
                return parsedate_to_datetime(value).strftime("%Y-%m-%d")
            except Exception:
                pass
    return ""


def extract_summary(entry) -> str:
    summary = clean_html(entry.get("summary", ""))
    if summary:
        return summary
    if entry.get("title"):
        return clean_html(entry["title"])
    return ""


def fetch_rows():
    rows = []
    seen = set()

    for query in QUERIES:
        for start_date, end_date in month_windows(YEAR):
            rss_url = build_rss_url(query, start_date, end_date)
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                headline = clean_html(entry.get("title", ""))
                url = entry.get("link", "").strip()
                date = parse_published(entry)
                summary = extract_summary(entry)

                if not headline or not url:
                    continue

                key = (headline.lower(), url)
                if key in seen:
                    continue
                seen.add(key)

                rows.append({
                    "date": date,
                    "headline": headline,
                    "url": url,
                    "summary": summary,
                    "search_query": query,
                    "window_start": start_date,
                    "window_end": end_date,
                    "rss_url": rss_url,
                })
            time.sleep(SLEEP_SECONDS)
    return rows


def main():
    rows = fetch_rows()
    df = pd.DataFrame(rows)

    if df.empty:
        print("No rows returned from Google News RSS.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "headline"], na_position="last").drop_duplicates(subset=["headline", "url"])

    # Keep the main dataset exactly in the shape you requested.
    final_df = df[["date", "headline", "url", "summary"]].copy()
    final_df["date"] = final_df["date"].dt.strftime("%Y-%m-%d")
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Optional audit file so you can see which search query/window produced each row.
    audit_df = df.copy()
    audit_df["date"] = audit_df["date"].dt.strftime("%Y-%m-%d")
    audit_df.to_csv("hsbc_google_news_rss_2021_audit.csv", index=False, encoding="utf-8-sig")

    print(f"Saved {len(final_df)} rows to {OUTPUT_CSV}")
    print("Also saved hsbc_google_news_rss_2021_audit.csv with query provenance.")


if __name__ == "__main__":
    main()
