-- IEOM execution plan tracking schema + seed data
-- Usage:
--   sqlite3 ieom_plan_tracking.db < ieom_plan_tracking.sql

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS phases (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    ordering INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    phase_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    details TEXT NOT NULL,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('todo', 'in_progress', 'blocked', 'done')),
    depends_on_code TEXT,
    owner TEXT NOT NULL DEFAULT 'ml_team',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (phase_id) REFERENCES phases(id)
);

CREATE TABLE IF NOT EXISTS status_log (
    id INTEGER PRIMARY KEY,
    todo_code TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    note TEXT,
    changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (todo_code) REFERENCES todos(code)
);

INSERT OR REPLACE INTO phases (id, code, name, description, ordering) VALUES
    (1, 'p0', 'PHASE 0', 'Lock problem statement, states, and metrics', 0),
    (2, 'p1', 'PHASE 1', 'Finalize model interface and validation', 1),
    (3, 'p2', 'PHASE 2', 'Handoff interface to MATLAB team', 2),
    (4, 'p3', 'PHASE 3', 'Integration collaboration', 3),
    (5, 'p4', 'PHASE 4', 'A/B experiment execution', 4),
    (6, 'p5', 'PHASE 5', 'Paper write-up', 5);

INSERT OR REPLACE INTO todos (code, phase_id, title, details, priority, status, depends_on_code, owner) VALUES
    ('p0-problem-statement', 1, 'Write 1-paragraph research question', 'Lock final problem statement for IEOM narrative.', 1, 'done', NULL, 'ml_team'),
    ('p0-state-definitions', 1, 'Define 6 hesitation states operationally', 'Freeze state taxonomy used by model and policy mapping.', 2, 'done', 'p0-problem-statement', 'ml_team'),
    ('p0-success-metrics', 1, 'Define success metrics', 'Finalize safety, efficiency, and quality KPIs.', 3, 'done', 'p0-state-definitions', 'ml_team'),

    ('p1-input-schema', 2, 'Freeze input schema', 'Finalize and document 7-feature input interface.', 1, 'done', 'p0-success-metrics', 'ml_team'),
    ('p1-output-schema', 2, 'Freeze output schema', 'Finalize prediction dataclass and JSON contract.', 2, 'done', 'p1-input-schema', 'ml_team'),
    ('p1-inference-wrapper', 2, 'Build inference wrapper', 'Provide deterministic predictor class and CLI interface.', 3, 'done', 'p1-output-schema', 'ml_team'),
    ('p1-model-validation', 2, 'Compute confusion matrix and F1', 'Run per-class and macro F1 metrics on validation/test splits.', 4, 'todo', 'p1-inference-wrapper', 'ml_team'),
    ('p1-error-analysis', 2, 'Audit 25 examples per class', 'Collect representative failures and likely causes.', 5, 'todo', 'p1-model-validation', 'ml_team'),

    ('p2-feature-schema-doc', 3, 'Document feature schema for MATLAB', 'Publish feature definitions, units, ranges, and edge cases.', 1, 'done', 'p1-output-schema', 'ml_team'),
    ('p2-policy-mapping', 3, 'Define state to action mapping', 'Map predicted state to robot behavior policy table.', 2, 'done', 'p2-feature-schema-doc', 'ml_team'),
    ('p2-integration-guide', 3, 'Write MATLAB integration guide', 'Document callable entrypoints and setup instructions.', 3, 'done', 'p2-policy-mapping', 'ml_team'),

    ('p3-weekly-checklist', 4, 'Create weekly integration checklist', 'Define recurring checkpoints for model-sim alignment.', 1, 'todo', 'p2-integration-guide', 'both'),
    ('p3-debug-interface', 4, 'Debug model to MATLAB interface', 'Resolve integration issues in simulator loop.', 2, 'todo', 'p3-weekly-checklist', 'both'),
    ('p3-validate-features', 4, 'Validate 7 extracted features', 'Spot-check extracted simulator features against schema.', 3, 'todo', 'p3-debug-interface', 'both'),
    ('p3-verify-outputs', 4, 'Verify outputs in 3 scenarios', 'Ensure model outputs align with expected scenario behavior.', 4, 'todo', 'p3-validate-features', 'both'),

    ('p4-scenarios', 5, 'Define 5 operator profiles', 'Finalize scenario matrix for A/B evaluation.', 1, 'todo', 'p3-verify-outputs', 'matlab_team'),
    ('p4-ab-setup', 5, 'Set up A/B infrastructure', 'Implement baseline vs hesitation-aware policy switching and logs.', 2, 'todo', 'p4-scenarios', 'matlab_team'),
    ('p4-run-trials', 5, 'Run 50 to 200 trials per scenario', 'Execute controlled trial batches for each profile.', 3, 'todo', 'p4-ab-setup', 'matlab_team'),
    ('p4-collect-metrics', 5, 'Collect safety/efficiency/quality metrics', 'Persist raw and aggregated metrics for analysis.', 4, 'todo', 'p4-run-trials', 'both'),
    ('p4-analyze-results', 5, 'Analyze comparative results', 'Create summary tables and plots across baseline vs policy.', 5, 'todo', 'p4-collect-metrics', 'ml_team'),

    ('p5-system-diagram', 6, 'Create system diagram', 'Visualize full pipeline and data flow.', 1, 'todo', 'p4-analyze-results', 'ml_team'),
    ('p5-results-section', 6, 'Draft results section', 'Write quantitative findings and interpretation.', 2, 'todo', 'p5-system-diagram', 'ml_team'),
    ('p5-error-analysis', 6, 'Write failure mode analysis', 'Document limitations, edge cases, and mitigations.', 3, 'todo', 'p5-results-section', 'ml_team'),
    ('p5-key-claim', 6, 'Finalize one-sentence claim', 'Lock concise contribution statement backed by results.', 4, 'todo', 'p5-error-analysis', 'ml_team');

INSERT INTO status_log (todo_code, from_status, to_status, note)
SELECT code, NULL, status, 'Seeded from SESSION_SUMMARY + PHASE_1_SUMMARY'
FROM todos
WHERE NOT EXISTS (SELECT 1 FROM status_log);
