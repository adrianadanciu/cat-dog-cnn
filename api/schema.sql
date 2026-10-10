--the table where the API saves every prediction
--it is also the queue
CREATE TABLE IF NOT EXISTS predictions (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename    TEXT NOT NULL,
    image       BYTEA,
    --pending/processing/done 
    status      TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    label       TEXT CHECK (label IN ('cat', 'dog')),
    confidence  REAL CHECK (confidence BETWEEN 0 AND 1),
    worker      TEXT,
    attempts    INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
