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
#the server puts the photo in the queue
def add_to_queue(filename: str, image: bytes) -> dict:
    with pool.connection() as conn:
        return conn.execute(
            """
            INSERT INTO predictions (filename, image)
            VALUES (%s, %s)
            RETURNING id, filename, status, label, confidence, worker, created_at
            """,
            (filename, image),
        ).fetchone()
#a worker takes the oldest photo that nobody is working on
def take_from_queue(worker: str) -> dict | None:
    with pool.connection() as conn:
        return conn.execute(
            """
            UPDATE predictions
            SET status = 'processing', worker = %s, started_at = now(), attempts = attempts + 1
            WHERE id = (
                SELECT id
                FROM predictions
                WHERE status = 'pending'
                   OR (status = 'processing' AND started_at < now() - interval '60 seconds')
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, image, attempts
            """,
            (worker,),
        ).fetchone()
#the photo itself is deleted, because it is not needed anymore
def save_result(prediction_id: int, worker: str, label: str, confidence: float):
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE predictions
            SET status = 'done', label = %s, confidence = %s, finished_at = now(), image = NULL
            WHERE id = %s AND worker = %s
            """,
            (label, confidence, prediction_id, worker),
        )
def mark_failed(prediction_id: int, worker: str):
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE predictions
            SET status = 'failed', finished_at = now(), image = NULL
            WHERE id = %s AND worker = %s
            """,
            (prediction_id, worker),
        )
def list_predictions(limit: int) -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(
            """
            SELECT id, filename, status, label, confidence, worker, created_at
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
            SELECT id, filename, status, label, confidence, worker, created_at
            FROM predictions
            WHERE id = %s
            """,
            (prediction_id,),
        ).fetchone()
#only the photos that are done have a label
def get_stats() -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(
            """
            SELECT label,
                   COUNT(*) AS count,
                   ROUND(AVG(confidence)::numeric, 3) AS avg_confidence
            FROM predictions
            WHERE status = 'done'
            GROUP BY label
            ORDER BY label
            """
        ).fetchall()
