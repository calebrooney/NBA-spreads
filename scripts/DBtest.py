import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Force load .env from repo root (one level above /scripts)
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

url = os.getenv("DATABASE_URL")
print("Loaded .env from:", ENV_PATH)
print("DATABASE_URL starts with:", (url or "")[:15])

if not url:
    raise ValueError("DATABASE_URL is missing. Your .env did not load or is malformed.")

if "psql" in url:
    raise ValueError("DATABASE_URL incorrectly contains the text 'psql'. Paste only the URL.")

conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute("SELECT current_database(), current_user, inet_server_addr();")
print(cur.fetchone())
conn.close()