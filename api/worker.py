import os
#socket is used for network communication
import socket
import time
from api import db, ml
#after 3 tries, a photo is marked as failed instead of being taken again
MAX_ATTEMPTS = 3
#the name shows which worker classified each photo
WORKER_NAME = f"{os.getpid()}@{socket.gethostname()}"
def main():
    db.pool.open(wait=True, timeout=10)
    db.create_schema()
    ml.load()
    print(f"Worker {WORKER_NAME} is waiting for photos")
    try:
        while True:
            job = db.take_from_queue(WORKER_NAME)
            if job is None:
                #the queue is empty, so ask again in a second
                time.sleep(1)
                continue
            if job["attempts"] > MAX_ATTEMPTS:
                db.mark_failed(job["id"], WORKER_NAME)
                print(f"Photo {job['id']}: failed too many times")
                continue
            try:
                label, confidence = ml.predict(job["image"])
            except Exception as error:
                db.mark_failed(job["id"], WORKER_NAME)
                print(f"Photo {job['id']}: failed ({error})")
                continue
            db.save_result(job["id"], WORKER_NAME, label, confidence)
            print(f"Photo {job['id']}: {label} ({confidence:.1%})")
    #Ctrl + C stops the worker
    except KeyboardInterrupt:
        print("Worker stopped")
    finally:
        db.pool.close()
if __name__ == "__main__":
    main()
