-- pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table — both trend signals and product chunks land here
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    source       TEXT NOT NULL CHECK (source IN ('trend', 'product')),
    external_id  TEXT,                          -- TrendRadar news_items.id or product slug
    title        TEXT,                          -- original (may be Chinese for trends)
    title_en     TEXT,                          -- English translation
    content      TEXT NOT NULL,                 -- original or chunked content
    content_en   TEXT,                          -- English version
    platform     TEXT,                          -- trends: baidu/weibo/etc; products: NULL
    rank         INTEGER,                       -- trends: original rank; products: NULL
    metadata     JSONB DEFAULT '{}'::jsonb,
    embedding    vector(1024),                  -- mistral-embed dim
    created_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE(source, external_id, content)
);

-- HNSW index for cosine similarity (mistral-embed uses L2/cosine)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Lookup indexes for filtering
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);
CREATE INDEX IF NOT EXISTS documents_platform_idx ON documents (platform) WHERE platform IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_metadata_gin ON documents USING gin (metadata);

-- Eval results table — for tracking eval runs over time (retention: keep all in eval_results/runs/ filesystem; this is a query convenience)
CREATE TABLE IF NOT EXISTS eval_runs (
    id           SERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL UNIQUE,
    model        TEXT NOT NULL,
    metrics      JSONB NOT NULL,
    notes        TEXT,
    created_at   TIMESTAMP DEFAULT NOW()
);
