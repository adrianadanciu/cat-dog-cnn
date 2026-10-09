--the table where the API saves every prediction.
CREATE TABLE IF NOT EXISTS predictions (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename   TEXT NOT NULL,
    label      TEXT NOT NULL CHECK (label IN ('cat', 'dog')),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
