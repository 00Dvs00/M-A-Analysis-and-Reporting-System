import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# --- DEBUG: Print loaded environment variables ---
print("--- Loaded Environment Variables ---")
print(f"DB_USER: '{DB_USER}'")
print(f"DB_PASSWORD: '{DB_PASSWORD}'")
print(f"DB_HOST: '{DB_HOST}'")
print(f"DB_PORT: '{DB_PORT}'")
print(f"DB_NAME: '{DB_NAME}'")
print("------------------------------------")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"Attempting to connect to: {DB_HOST}:{DB_PORT} as {DB_USER}")

try:
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("Connection successful!")
    connection.close()
except Exception as e:
    print(f"Connection failed: {e}")
