-- Immutable synthetic replay packets for Desk. Apply after 001_work.sql.
BEGIN;
SET LOCAL search_path = pg_catalog, pg_temp;

CREATE TABLE studio.bd_packets (
    tenant_id uuid NOT NULL,
    work_item_id uuid NOT NULL,
    run_id uuid NOT NULL,
    packet_index integer NOT NULL CHECK (packet_index >= 0),
    mode text NOT NULL CHECK (mode = 'replay'),
    source_kind text NOT NULL CHECK (source_kind = 'synthetic_fixture'),
    segment text NOT NULL,
    organization text NOT NULL CHECK (btrim(organization) <> ''),
    source_url text NOT NULL,
    source_quote text NOT NULL,
    signals text[] NOT NULL,
    insights text[] NOT NULL,
    emails text[] NOT NULL,
    phones text[] NOT NULL,
    summary text NOT NULL,
    disposition text NOT NULL,
    candidate_status text NOT NULL CHECK (candidate_status IN ('supported', 'inconclusive', 'rejected')),
    candidate_reason text NOT NULL,
    completed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, run_id, packet_index),
    FOREIGN KEY (tenant_id, work_item_id, run_id)
        REFERENCES studio.runs (tenant_id, work_item_id, id)
);
CREATE INDEX bd_packets_for_desk ON studio.bd_packets
    (tenant_id, completed_at DESC, run_id, packet_index) WHERE candidate_status = 'supported';

-- Earlier work receipts have no packet identity. For packet decisions, the
-- existing run FK fixes the work item and this FK fixes its exact packet.
ALTER TABLE studio.decision_receipts
    ADD COLUMN packet_index integer,
    ADD CHECK (packet_index IS NULL OR
        (run_id IS NOT NULL AND decision IN ('research_next', 'watch', 'dismissed'))),
    ADD FOREIGN KEY (tenant_id, run_id, packet_index)
        REFERENCES studio.bd_packets (tenant_id, run_id, packet_index);
CREATE UNIQUE INDEX decision_receipts_one_per_packet ON studio.decision_receipts
    (tenant_id, run_id, packet_index) WHERE packet_index IS NOT NULL;
CREATE INDEX decision_receipts_for_record ON studio.decision_receipts
    (tenant_id, recorded_at DESC, id DESC) WHERE packet_index IS NOT NULL;

ALTER TABLE studio.bd_packets ENABLE ROW LEVEL SECURITY;
ALTER TABLE studio.bd_packets FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_access ON studio.bd_packets TO aiia_studio_app
    USING (tenant_id = (SELECT studio.current_tenant_id()))
    WITH CHECK (tenant_id = (SELECT studio.current_tenant_id()));
REVOKE ALL ON studio.bd_packets FROM PUBLIC, aiia_studio_app;
GRANT SELECT, INSERT ON studio.bd_packets TO aiia_studio_app;
COMMIT;
