CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content             TEXT NOT NULL,
    embedding           vector(768),
    token_count         INTEGER,
    metadata            JSONB NOT NULL DEFAULT '{}',
    document_id         TEXT NOT NULL,
    document_type       TEXT NOT NULL,
    chapter             TEXT,
    article             TEXT,
    paragraph           TEXT,
    contains_table      BOOLEAN DEFAULT FALSE,
    contains_penalty    BOOLEAN DEFAULT FALSE,
    is_definition       BOOLEAN DEFAULT FALSE,
    jurisdiction        TEXT DEFAULT 'EU',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx   ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_document_type_idx ON document_chunks (document_type);
CREATE INDEX IF NOT EXISTS chunks_article_idx       ON document_chunks (article);
CREATE INDEX IF NOT EXISTS chunks_penalty_idx       ON document_chunks (contains_penalty);

CREATE OR REPLACE VIEW corpus_stats AS
SELECT
    document_id,
    document_type,
    COUNT(*)                                          AS chunk_count,
    COUNT(*) FILTER (WHERE contains_table)            AS table_chunks,
    COUNT(*) FILTER (WHERE embedding IS NOT NULL)     AS embedded_chunks
FROM document_chunks
GROUP BY document_id, document_type
ORDER BY chunk_count DESC;
