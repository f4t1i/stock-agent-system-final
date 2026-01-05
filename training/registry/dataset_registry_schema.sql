-- Dataset Registry Postgres Schema - Task 2.1
-- 
-- Complete database schema for dataset registry system
-- 
-- Tables:
-- 1. datasets - Main dataset metadata
-- 2. dataset_versions - Version management with semantic versioning
-- 3. dataset_lineage - Parent-child relationships and provenance
-- 4. dataset_quality_metrics - Quality and judge scores
--
-- Features:
-- - Semantic versioning (MAJOR.MINOR.PATCH)
-- - SHA256 integrity checking
-- - Lineage tracking (parent versions, backtest sources)
-- - Quality metrics (avg/min/max scores)
-- - Judge evaluation scores
-- - Full-text search on descriptions
-- - Timestamp tracking (created/updated)
-- - Soft delete support
--
-- Phase A1 Week 3-4: Task 2.1 COMPLETE

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For full-text search

-- ============================================================================
-- Table 1: datasets
-- Main dataset metadata table
-- ============================================================================

CREATE TABLE IF NOT EXISTS datasets (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Dataset identification
    agent_type VARCHAR(50) NOT NULL, -- technical, news, fundamental, strategist, supervisor
    name VARCHAR(255) NOT NULL, -- Human-readable name
    description TEXT, -- Optional description
    
    -- Format and type
    format VARCHAR(20) NOT NULL, -- chatml, alpaca
    dataset_type VARCHAR(50) DEFAULT 'sft', -- sft, rl, mixed
    
    -- Current version (denormalized for quick access)
    current_version VARCHAR(20), -- e.g., "1.2.3"
    current_version_id UUID, -- FK to dataset_versions
    
    -- Statistics (denormalized from latest version)
    total_examples INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- active, archived, deprecated
    is_deleted BOOLEAN DEFAULT FALSE, -- Soft delete
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT datasets_agent_type_check CHECK (
        agent_type IN ('technical', 'news', 'fundamental', 'strategist_junior', 'strategist_senior', 'supervisor')
    ),
    CONSTRAINT datasets_format_check CHECK (
        format IN ('chatml', 'alpaca')
    ),
    CONSTRAINT datasets_status_check CHECK (
        status IN ('active', 'archived', 'deprecated')
    ),
    CONSTRAINT datasets_unique_agent_type UNIQUE (agent_type)
);

-- Indexes for datasets
CREATE INDEX idx_datasets_agent_type ON datasets(agent_type);
CREATE INDEX idx_datasets_status ON datasets(status);
CREATE INDEX idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX idx_datasets_name_trgm ON datasets USING gin(name gin_trgm_ops); -- Full-text search

-- ============================================================================
-- Table 2: dataset_versions
-- Version management with semantic versioning
-- ============================================================================

CREATE TABLE IF NOT EXISTS dataset_versions (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Foreign key to datasets
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    
    -- Semantic versioning
    version VARCHAR(20) NOT NULL, -- e.g., "1.2.3"
    major_version INTEGER NOT NULL,
    minor_version INTEGER NOT NULL,
    patch_version INTEGER NOT NULL,
    
    -- File information
    file_path TEXT NOT NULL, -- Path to dataset file
    file_size_bytes BIGINT NOT NULL,
    sha256_hash VARCHAR(64) NOT NULL, -- SHA256 for integrity
    
    -- Content statistics
    example_count INTEGER NOT NULL,
    avg_tokens_per_example FLOAT,
    total_tokens BIGINT,
    
    -- Quality metrics (summary from dataset_quality_metrics)
    avg_quality_score FLOAT,
    min_quality_score FLOAT,
    max_quality_score FLOAT,
    
    -- Judge evaluation (summary)
    avg_judge_score FLOAT,
    judge_pass_rate FLOAT, -- Percentage passing judge threshold
    judge_evaluated_count INTEGER DEFAULT 0,
    
    -- Metadata
    description TEXT,
    changelog TEXT, -- What changed in this version
    tags TEXT[], -- Array of tags for categorization
    
    -- Status
    is_latest BOOLEAN DEFAULT TRUE, -- Is this the latest version?
    is_deleted BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100), -- User or system that created this version
    
    -- Constraints
    CONSTRAINT dataset_versions_unique_version UNIQUE (dataset_id, version),
    CONSTRAINT dataset_versions_version_format CHECK (
        version ~ '^\d+\.\d+\.\d+$' -- Regex for semantic versioning
    ),
    CONSTRAINT dataset_versions_positive_counts CHECK (
        example_count > 0 AND file_size_bytes > 0
    )
);

-- Indexes for dataset_versions
CREATE INDEX idx_dataset_versions_dataset_id ON dataset_versions(dataset_id);
CREATE INDEX idx_dataset_versions_version ON dataset_versions(version);
CREATE INDEX idx_dataset_versions_is_latest ON dataset_versions(is_latest) WHERE is_latest = TRUE;
CREATE INDEX idx_dataset_versions_created_at ON dataset_versions(created_at DESC);
CREATE INDEX idx_dataset_versions_sha256 ON dataset_versions(sha256_hash);
CREATE INDEX idx_dataset_versions_tags ON dataset_versions USING gin(tags); -- Array search

-- ============================================================================
-- Table 3: dataset_lineage
-- Parent-child relationships and provenance tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS dataset_lineage (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Child version (the derived dataset)
    child_version_id UUID NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    
    -- Parent version (the source dataset)
    parent_version_id UUID REFERENCES dataset_versions(id) ON DELETE SET NULL,
    
    -- Backtest source (if derived from backtest)
    backtest_id VARCHAR(100),
    backtest_date TIMESTAMP WITH TIME ZONE,
    
    -- Transformation information
    transformation_type VARCHAR(50), -- filter, merge, augment, retrain
    transformation_params JSONB, -- Parameters used in transformation
    
    -- Statistics
    examples_inherited INTEGER DEFAULT 0, -- How many examples came from parent
    examples_new INTEGER DEFAULT 0, -- How many new examples added
    examples_modified INTEGER DEFAULT 0, -- How many examples modified
    examples_removed INTEGER DEFAULT 0, -- How many examples removed
    
    -- Metadata
    notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT dataset_lineage_transformation_check CHECK (
        transformation_type IN ('filter', 'merge', 'augment', 'retrain', 'judge_filter', 'quality_filter')
    )
);

-- Indexes for dataset_lineage
CREATE INDEX idx_dataset_lineage_child ON dataset_lineage(child_version_id);
CREATE INDEX idx_dataset_lineage_parent ON dataset_lineage(parent_version_id);
CREATE INDEX idx_dataset_lineage_backtest ON dataset_lineage(backtest_id);
CREATE INDEX idx_dataset_lineage_transformation ON dataset_lineage(transformation_type);

-- ============================================================================
-- Table 4: dataset_quality_metrics
-- Detailed quality and judge scores per example
-- ============================================================================

CREATE TABLE IF NOT EXISTS dataset_quality_metrics (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Foreign key to dataset version
    version_id UUID NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    
    -- Example identification
    example_index INTEGER NOT NULL, -- Index in dataset file
    trajectory_id VARCHAR(100), -- Original trajectory ID (if applicable)
    
    -- Quality scores (from QualityScorer)
    quality_score FLOAT NOT NULL,
    reward_score FLOAT,
    confidence_score FLOAT,
    reasoning_score FLOAT,
    consistency_score FLOAT,
    
    -- Judge evaluation (from LLMJudge)
    judge_score FLOAT,
    judge_passed BOOLEAN,
    judge_feedback TEXT,
    judge_evaluated_at TIMESTAMP WITH TIME ZONE,
    
    -- Example metadata
    symbol VARCHAR(20), -- Stock symbol (if applicable)
    agent_type VARCHAR(50), -- Agent that generated this example
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT dataset_quality_metrics_unique_example UNIQUE (version_id, example_index),
    CONSTRAINT dataset_quality_metrics_scores_range CHECK (
        quality_score >= 0 AND quality_score <= 1 AND
        (reward_score IS NULL OR (reward_score >= 0 AND reward_score <= 1)) AND
        (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)) AND
        (reasoning_score IS NULL OR (reasoning_score >= 0 AND reasoning_score <= 1)) AND
        (consistency_score IS NULL OR (consistency_score >= 0 AND consistency_score <= 1)) AND
        (judge_score IS NULL OR (judge_score >= 0 AND judge_score <= 1))
    )
);

-- Indexes for dataset_quality_metrics
CREATE INDEX idx_dataset_quality_metrics_version ON dataset_quality_metrics(version_id);
CREATE INDEX idx_dataset_quality_metrics_quality_score ON dataset_quality_metrics(quality_score DESC);
CREATE INDEX idx_dataset_quality_metrics_judge_score ON dataset_quality_metrics(judge_score DESC);
CREATE INDEX idx_dataset_quality_metrics_judge_passed ON dataset_quality_metrics(judge_passed) WHERE judge_passed = TRUE;
CREATE INDEX idx_dataset_quality_metrics_trajectory ON dataset_quality_metrics(trajectory_id);
CREATE INDEX idx_dataset_quality_metrics_symbol ON dataset_quality_metrics(symbol);

-- ============================================================================
-- Triggers and Functions
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for datasets table
CREATE TRIGGER trigger_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to update dataset statistics when version is added
CREATE OR REPLACE FUNCTION update_dataset_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update current_version and stats in datasets table
    UPDATE datasets
    SET 
        current_version = NEW.version,
        current_version_id = NEW.id,
        total_examples = NEW.example_count,
        total_size_bytes = NEW.file_size_bytes,
        updated_at = NOW()
    WHERE id = NEW.dataset_id AND NEW.is_latest = TRUE;
    
    -- Mark other versions as not latest
    UPDATE dataset_versions
    SET is_latest = FALSE
    WHERE dataset_id = NEW.dataset_id 
        AND id != NEW.id 
        AND is_latest = TRUE;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for dataset_versions table
CREATE TRIGGER trigger_update_dataset_stats
    AFTER INSERT OR UPDATE ON dataset_versions
    FOR EACH ROW
    WHEN (NEW.is_latest = TRUE)
    EXECUTE FUNCTION update_dataset_stats();

-- Function to calculate quality metrics summary
CREATE OR REPLACE FUNCTION calculate_quality_summary(p_version_id UUID)
RETURNS TABLE (
    avg_quality FLOAT,
    min_quality FLOAT,
    max_quality FLOAT,
    avg_judge FLOAT,
    judge_pass_rate FLOAT,
    judge_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(quality_score)::FLOAT,
        MIN(quality_score)::FLOAT,
        MAX(quality_score)::FLOAT,
        AVG(judge_score)::FLOAT,
        (COUNT(*) FILTER (WHERE judge_passed = TRUE)::FLOAT / 
         NULLIF(COUNT(*) FILTER (WHERE judge_score IS NOT NULL), 0))::FLOAT,
        COUNT(*) FILTER (WHERE judge_score IS NOT NULL)::INTEGER
    FROM dataset_quality_metrics
    WHERE version_id = p_version_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View: Latest versions of all datasets
CREATE OR REPLACE VIEW v_latest_datasets AS
SELECT 
    d.id AS dataset_id,
    d.agent_type,
    d.name,
    d.description,
    d.format,
    d.status,
    dv.id AS version_id,
    dv.version,
    dv.example_count,
    dv.file_size_bytes,
    dv.avg_quality_score,
    dv.avg_judge_score,
    dv.judge_pass_rate,
    dv.created_at AS version_created_at,
    d.created_at AS dataset_created_at
FROM datasets d
LEFT JOIN dataset_versions dv ON d.current_version_id = dv.id
WHERE d.is_deleted = FALSE AND d.status = 'active';

-- View: Dataset lineage tree
CREATE OR REPLACE VIEW v_dataset_lineage_tree AS
SELECT 
    dl.id AS lineage_id,
    dl.child_version_id,
    child_dv.version AS child_version,
    child_d.agent_type AS child_agent_type,
    dl.parent_version_id,
    parent_dv.version AS parent_version,
    parent_d.agent_type AS parent_agent_type,
    dl.backtest_id,
    dl.transformation_type,
    dl.examples_inherited,
    dl.examples_new,
    dl.examples_modified,
    dl.examples_removed,
    dl.created_at
FROM dataset_lineage dl
LEFT JOIN dataset_versions child_dv ON dl.child_version_id = child_dv.id
LEFT JOIN datasets child_d ON child_dv.dataset_id = child_d.id
LEFT JOIN dataset_versions parent_dv ON dl.parent_version_id = parent_dv.id
LEFT JOIN datasets parent_d ON parent_dv.dataset_id = parent_d.id;

-- View: Quality metrics summary by version
CREATE OR REPLACE VIEW v_quality_metrics_summary AS
SELECT 
    dqm.version_id,
    dv.version,
    d.agent_type,
    COUNT(*) AS total_examples,
    AVG(dqm.quality_score) AS avg_quality_score,
    MIN(dqm.quality_score) AS min_quality_score,
    MAX(dqm.quality_score) AS max_quality_score,
    AVG(dqm.judge_score) AS avg_judge_score,
    COUNT(*) FILTER (WHERE dqm.judge_passed = TRUE) AS judge_passed_count,
    COUNT(*) FILTER (WHERE dqm.judge_score IS NOT NULL) AS judge_evaluated_count,
    (COUNT(*) FILTER (WHERE dqm.judge_passed = TRUE)::FLOAT / 
     NULLIF(COUNT(*) FILTER (WHERE dqm.judge_score IS NOT NULL), 0)) AS judge_pass_rate
FROM dataset_quality_metrics dqm
JOIN dataset_versions dv ON dqm.version_id = dv.id
JOIN datasets d ON dv.dataset_id = d.id
GROUP BY dqm.version_id, dv.version, d.agent_type;

-- ============================================================================
-- Sample Data (for testing)
-- ============================================================================

-- Insert sample dataset
INSERT INTO datasets (agent_type, name, description, format, status)
VALUES 
    ('technical', 'Technical Analysis Dataset', 'Dataset for technical analysis agent training', 'chatml', 'active'),
    ('news', 'News Sentiment Dataset', 'Dataset for news analysis agent training', 'chatml', 'active')
ON CONFLICT (agent_type) DO NOTHING;

-- ============================================================================
-- Grants (adjust as needed for your user)
-- ============================================================================

-- Grant permissions to application user (replace 'app_user' with actual username)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO app_user;

-- ============================================================================
-- Comments (documentation)
-- ============================================================================

COMMENT ON TABLE datasets IS 'Main dataset metadata table with current version info';
COMMENT ON TABLE dataset_versions IS 'Version history with semantic versioning and file integrity';
COMMENT ON TABLE dataset_lineage IS 'Parent-child relationships and provenance tracking';
COMMENT ON TABLE dataset_quality_metrics IS 'Per-example quality and judge evaluation scores';

COMMENT ON COLUMN datasets.current_version IS 'Denormalized current version for quick access';
COMMENT ON COLUMN dataset_versions.sha256_hash IS 'SHA256 hash for file integrity verification';
COMMENT ON COLUMN dataset_versions.is_latest IS 'Flag indicating if this is the latest version';
COMMENT ON COLUMN dataset_lineage.transformation_type IS 'Type of transformation applied: filter, merge, augment, retrain';
COMMENT ON COLUMN dataset_quality_metrics.judge_passed IS 'Whether example passed LLM judge evaluation';

-- ============================================================================
-- Schema Version
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description)
VALUES ('2.1.0', 'Initial dataset registry schema with 4 tables')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- End of Schema
-- ============================================================================

-- Phase A1 Week 3-4: Task 2.1 COMPLETE
