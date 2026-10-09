#os = operating system
import os
from pathlib import Path
#psycopg is the library throughout which the code talks to the database
#it sets the rows as dictionaries
#a dictionary is like an unordered map
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
#where the database is
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/catdog")
#path to schema.sql
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
#the pool keeps a few connections open and lends them to the requests
pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=5,
    #kwargs = keywords arguments
    kwargs={"row_factory": dict_row},
    open=False,
)
def create_schema():
    with pool.connection() as conn:
        conn.execute(SCHEMA_PATH.read_text())
#%s marks a value that psycopg sends separately from the SQL
def insert_prediction(filename: str, label: str, confidence: float) -> dict:
    with pool.connection() as conn:
        return conn.execute(
            """
            INSERT INTO predictions (filename, label, confidence)
            VALUES (%s, %s, %s)
            RETURNING id, filename, label, confidence, created_at
            """,
            (filename, label, confidence),
        ).fetchone()
def list_predictions(limit: int) -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(
            """
            SELECT id, filename, label, confidence, created_at
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
def get_prediction(prediction_id: int) -> dict | None:
    with pool.connection() as conn:
        return conn.execute(
            """
            SELECT id, filename, label, confidence, created_at
            FROM predictions
            WHERE id = %s
            """,
            (prediction_id,),
        ).fetchone()
def get_stats() -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(
            """
            SELECT label,
                   COUNT(*) AS count,
                   ROUND(AVG(confidence)::numeric, 3) AS avg_confidence
            FROM predictions
            GROUP BY label
            ORDER BY label
            """
        ).fetchall()
