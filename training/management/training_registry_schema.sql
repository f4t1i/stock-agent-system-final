-- Training Job Registry Schema - Task 7.1
-- 
-- Centralized training job tracking for all providers (OpenAI, Anthropic, etc.)
--
-- Tables:
-- - training_jobs: Main job metadata
-- - training_hyperparameters: Hyperparameter configurations
-- - training_metrics: Training metrics over time
-- - training_models: Fine-tuned model registry
--
-- Phase A1 Week 5-6: Task 7.1 COMPLETE

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable timestamp functions
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- =============================================================================
-- Main Tables
-- =============================================================================

-- Training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    -- Identity
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider VARCHAR(50) NOT NULL,  -- 'openai', 'anthropic', etc.
    provider_job_id VARCHAR(255) NOT NULL,  -- Provider's job ID
    
    -- Configuration
    base_model VARCHAR(255) NOT NULL,
    training_file_id VARCHAR(255) NOT NULL,
    validation_file_id VARCHAR(255),
    
    -- Status
    status VARCHAR(50) NOT NULL,  -- 'queued', 'running', 'succeeded', 'failed', 'cancelled'
    error_message TEXT,
    
    -- Results
    fine_tuned_model VARCHAR(255),
    trained_tokens BIGINT,
    
    -- Metadata
    created_by VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(provider, provider_job_id),
    CHECK (status IN ('validating', 'queued', 'running', 'succeeded', 'failed', 'cancelled'))
);

-- Training hyperparameters table
CREATE TABLE IF NOT EXISTS training_hyperparameters (
    -- Identity
    hyperparameter_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES training_jobs(job_id) ON DELETE CASCADE,
    
    -- Hyperparameters (provider-specific)
    n_epochs INTEGER,
    batch_size INTEGER,
    learning_rate_multiplier DECIMAL(10, 6),
    
    -- Additional parameters (JSONB for flexibility)
    additional_params JSONB DEFAULT '{}',
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(job_id)
);

-- Training metrics table
CREATE TABLE IF NOT EXISTS training_metrics (
    -- Identity
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES training_jobs(job_id) ON DELETE CASCADE,
    
    -- Metrics
    step INTEGER NOT NULL,
    epoch DECIMAL(10, 4),
    training_loss DECIMAL(10, 6),
    validation_loss DECIMAL(10, 6),
    training_accuracy DECIMAL(10, 6),
    validation_accuracy DECIMAL(10, 6),
    
    -- Additional metrics (JSONB for flexibility)
    additional_metrics JSONB DEFAULT '{}',
    
    -- Metadata
    recorded_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(job_id, step)
);

-- Training models table (fine-tuned model registry)
CREATE TABLE IF NOT EXISTS training_models (
    -- Identity
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES training_jobs(job_id) ON DELETE CASCADE,
    
    -- Model info
    model_name VARCHAR(255) NOT NULL UNIQUE,
    base_model VARCHAR(255) NOT NULL,
    provider VARCHAR(50) NOT NULL,
    
    -- Versioning
    version VARCHAR(50) NOT NULL,
    parent_model_id UUID REFERENCES training_models(model_id),
    
    -- Status
    status VARCHAR(50) NOT NULL,  -- 'active', 'deprecated', 'archived'
    
    -- Performance
    final_training_loss DECIMAL(10, 6),
    final_validation_loss DECIMAL(10, 6),
    trained_tokens BIGINT,
    
    -- Metadata
    description TEXT,
    tags TEXT[],
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    deprecated_at TIMESTAMP,
    
    -- Constraints
    CHECK (status IN ('active', 'deprecated', 'archived'))
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Training jobs indexes
CREATE INDEX IF NOT EXISTS idx_training_jobs_provider ON training_jobs(provider);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON training_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_base_model ON training_jobs(base_model);
CREATE INDEX IF NOT EXISTS idx_training_jobs_fine_tuned_model ON training_jobs(fine_tuned_model);

-- Training metrics indexes
CREATE INDEX IF NOT EXISTS idx_training_metrics_job_id ON training_metrics(job_id);
CREATE INDEX IF NOT EXISTS idx_training_metrics_step ON training_metrics(job_id, step);
CREATE INDEX IF NOT EXISTS idx_training_metrics_recorded_at ON training_metrics(recorded_at DESC);

-- Training models indexes
CREATE INDEX IF NOT EXISTS idx_training_models_provider ON training_models(provider);
CREATE INDEX IF NOT EXISTS idx_training_models_status ON training_models(status);
CREATE INDEX IF NOT EXISTS idx_training_models_base_model ON training_models(base_model);
CREATE INDEX IF NOT EXISTS idx_training_models_version ON training_models(version);
CREATE INDEX IF NOT EXISTS idx_training_models_parent ON training_models(parent_model_id);
CREATE INDEX IF NOT EXISTS idx_training_models_tags ON training_models USING GIN(tags);

-- =============================================================================
-- Views
-- =============================================================================

-- View: Active training jobs
CREATE OR REPLACE VIEW v_active_training_jobs AS
SELECT 
    j.job_id,
    j.provider,
    j.provider_job_id,
    j.base_model,
    j.status,
    j.fine_tuned_model,
    j.trained_tokens,
    j.created_at,
    j.started_at,
    EXTRACT(EPOCH FROM (NOW() - j.started_at)) AS elapsed_seconds,
    h.n_epochs,
    h.batch_size,
    h.learning_rate_multiplier
FROM training_jobs j
LEFT JOIN training_hyperparameters h ON j.job_id = h.job_id
WHERE j.status IN ('validating', 'queued', 'running')
ORDER BY j.created_at DESC;

-- View: Completed training jobs with metrics
CREATE OR REPLACE VIEW v_completed_training_jobs AS
SELECT 
    j.job_id,
    j.provider,
    j.provider_job_id,
    j.base_model,
    j.fine_tuned_model,
    j.trained_tokens,
    j.created_at,
    j.finished_at,
    EXTRACT(EPOCH FROM (j.finished_at - j.started_at)) AS training_duration_seconds,
    h.n_epochs,
    h.batch_size,
    h.learning_rate_multiplier,
    (SELECT training_loss FROM training_metrics WHERE job_id = j.job_id ORDER BY step DESC LIMIT 1) AS final_training_loss,
    (SELECT validation_loss FROM training_metrics WHERE job_id = j.job_id ORDER BY step DESC LIMIT 1) AS final_validation_loss
FROM training_jobs j
LEFT JOIN training_hyperparameters h ON j.job_id = h.job_id
WHERE j.status = 'succeeded'
ORDER BY j.finished_at DESC;

-- View: Model lineage
CREATE OR REPLACE VIEW v_model_lineage AS
WITH RECURSIVE lineage AS (
    -- Base case: models without parents
    SELECT 
        model_id,
        model_name,
        base_model,
        parent_model_id,
        version,
        1 AS generation,
        ARRAY[model_id] AS path
    FROM training_models
    WHERE parent_model_id IS NULL
    
    UNION ALL
    
    -- Recursive case: models with parents
    SELECT 
        m.model_id,
        m.model_name,
        m.base_model,
        m.parent_model_id,
        m.version,
        l.generation + 1,
        l.path || m.model_id
    FROM training_models m
    INNER JOIN lineage l ON m.parent_model_id = l.model_id
)
SELECT * FROM lineage
ORDER BY generation, model_name;

-- =============================================================================
-- Functions
-- =============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate training duration
CREATE OR REPLACE FUNCTION calculate_training_duration(p_job_id UUID)
RETURNS INTERVAL AS $$
DECLARE
    v_duration INTERVAL;
BEGIN
    SELECT finished_at - started_at INTO v_duration
    FROM training_jobs
    WHERE job_id = p_job_id;
    
    RETURN v_duration;
END;
$$ LANGUAGE plpgsql;

-- Function: Get latest metrics
CREATE OR REPLACE FUNCTION get_latest_metrics(p_job_id UUID)
RETURNS TABLE(
    step INTEGER,
    training_loss DECIMAL,
    validation_loss DECIMAL,
    training_accuracy DECIMAL,
    validation_accuracy DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.step,
        m.training_loss,
        m.validation_loss,
        m.training_accuracy,
        m.validation_accuracy
    FROM training_metrics m
    WHERE m.job_id = p_job_id
    ORDER BY m.step DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function: Get model lineage depth
CREATE OR REPLACE FUNCTION get_model_lineage_depth(p_model_id UUID)
RETURNS INTEGER AS $$
DECLARE
    v_depth INTEGER := 0;
    v_current_id UUID := p_model_id;
BEGIN
    WHILE v_current_id IS NOT NULL LOOP
        v_depth := v_depth + 1;
        SELECT parent_model_id INTO v_current_id
        FROM training_models
        WHERE model_id = v_current_id;
    END LOOP;
    
    RETURN v_depth;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Triggers
-- =============================================================================

-- Trigger: Update updated_at on training_jobs
CREATE TRIGGER trigger_training_jobs_updated_at
BEFORE UPDATE ON training_jobs
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Sample Queries
-- =============================================================================

-- Query 1: Get all active jobs
-- SELECT * FROM v_active_training_jobs;

-- Query 2: Get completed jobs with final metrics
-- SELECT * FROM v_completed_training_jobs;

-- Query 3: Get model lineage
-- SELECT * FROM v_model_lineage WHERE model_name = 'my-model';

-- Query 4: Get training metrics over time
-- SELECT step, training_loss, validation_loss
-- FROM training_metrics
-- WHERE job_id = 'xxx'
-- ORDER BY step;

-- Query 5: Get job statistics by provider
-- SELECT 
--     provider,
--     COUNT(*) AS total_jobs,
--     COUNT(*) FILTER (WHERE status = 'succeeded') AS successful_jobs,
--     COUNT(*) FILTER (WHERE status = 'failed') AS failed_jobs,
--     AVG(trained_tokens) FILTER (WHERE status = 'succeeded') AS avg_trained_tokens
-- FROM training_jobs
-- GROUP BY provider;
