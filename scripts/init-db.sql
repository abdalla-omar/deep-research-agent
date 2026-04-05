CREATE TABLE IF NOT EXISTS research_sessions (
  session_id TEXT PRIMARY KEY,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  status TEXT NOT NULL DEFAULT 'running',
  ledger JSONB NOT NULL,
  working_memory JSONB NOT NULL DEFAULT '[]'::jsonb,
  summary_memory TEXT NOT NULL DEFAULT '',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS evidence_cards (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  subquestion_id TEXT NOT NULL,
  claim_candidate TEXT NOT NULL,
  supporting_excerpt TEXT NOT NULL,
  source_url TEXT NOT NULL,
  source_title TEXT,
  retrieval_score DOUBLE PRECISION NOT NULL,
  source_quality DOUBLE PRECISION NOT NULL DEFAULT 0,
  recency_score DOUBLE PRECISION NOT NULL DEFAULT 0,
  novelty_score DOUBLE PRECISION NOT NULL DEFAULT 0,
  estimated_tokens INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_cards_session ON evidence_cards(session_id);
CREATE INDEX IF NOT EXISTS idx_evidence_cards_subq ON evidence_cards(subquestion_id);
