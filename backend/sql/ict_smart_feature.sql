-- ORACLE smart-feature table (Assignment 4, Task 4.3 / Appendix C)
-- Adds product-usage telemetry on top of the Assignment 3 schema.
-- SQLite dialect; safe to run once on a fresh database.

CREATE TABLE IF NOT EXISTS usage_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
    event_type    VARCHAR(32) NOT NULL,   -- run | api_call | login | register
    symbol        VARCHAR(24),
    latency_ms    REAL,
    payload_json  JSON NOT NULL DEFAULT '{}',
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_usage_events_event_type ON usage_events (event_type);
CREATE INDEX IF NOT EXISTS ix_usage_events_symbol     ON usage_events (symbol);
CREATE INDEX IF NOT EXISTS ix_usage_events_created_at ON usage_events (created_at);

-- Demo query used in the report: usage by event type + average latency
SELECT
    event_type,
    COUNT(*)                 AS events,
    ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
    COUNT(DISTINCT symbol)    AS distinct_symbols
FROM usage_events
GROUP BY event_type
ORDER BY events DESC;
