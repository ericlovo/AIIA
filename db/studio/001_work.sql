-- One-time migration for a dedicated Studio database. Run as the migration owner.
BEGIN;
SET LOCAL search_path = pg_catalog, pg_temp;

CREATE ROLE aiia_studio_app NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
CREATE SCHEMA studio;
REVOKE ALL ON SCHEMA studio FROM PUBLIC;
GRANT USAGE ON SCHEMA studio TO aiia_studio_app;

-- Only the migration owner provisions these mappings. Checking both values makes
-- a dropped/recreated or renamed login fail closed until deliberately rebound.
CREATE TABLE studio.tenant_principals (
    role_oid oid PRIMARY KEY,
    role_name name NOT NULL UNIQUE,
    tenant_id uuid NOT NULL
);

CREATE FUNCTION studio.current_tenant_id() RETURNS uuid
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
    SELECT principal.tenant_id
    FROM studio.tenant_principals AS principal
    JOIN pg_catalog.pg_roles AS login
      ON login.oid = principal.role_oid AND login.rolname = principal.role_name
    WHERE login.rolname = SESSION_USER AND login.rolcanlogin
$$;
REVOKE ALL ON FUNCTION studio.current_tenant_id() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION studio.current_tenant_id() TO aiia_studio_app;

CREATE TABLE studio.agents (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    name text NOT NULL CHECK (btrim(name) <> ''),
    PRIMARY KEY (tenant_id, id)
);

-- Enum ordering allows skipped levels while ruling out parent cycles, even
-- across concurrent writes. The FK verifies the parent's actual kind.
CREATE TYPE studio.work_kind AS ENUM ('epic', 'milestone', 'feature', 'assignment');
CREATE TABLE studio.work_items (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    kind studio.work_kind NOT NULL,
    title text NOT NULL CHECK (btrim(title) <> ''),
    status text NOT NULL DEFAULT 'planned' CHECK (status IN
        ('planned', 'pending', 'queued', 'running', 'in_progress', 'blocked', 'completed', 'failed', 'cancelled')),
    blocked_reason text,
    parent_id uuid,
    parent_kind studio.work_kind,
    agent_id uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, id, kind),
    CHECK ((parent_id IS NULL) = (parent_kind IS NULL)),
    CHECK (parent_kind < kind),
    FOREIGN KEY (tenant_id, parent_id, parent_kind) REFERENCES studio.work_items (tenant_id, id, kind),
    FOREIGN KEY (tenant_id, agent_id) REFERENCES studio.agents (tenant_id, id)
);
CREATE INDEX work_items_by_parent ON studio.work_items (tenant_id, parent_id);
CREATE INDEX work_items_by_status ON studio.work_items (tenant_id, status);

CREATE TABLE studio.runs (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    work_item_id uuid NOT NULL,
    agent_id uuid NOT NULL,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, work_item_id, id),
    FOREIGN KEY (tenant_id, work_item_id) REFERENCES studio.work_items (tenant_id, id),
    FOREIGN KEY (tenant_id, agent_id) REFERENCES studio.agents (tenant_id, id)
);

CREATE TABLE studio.decision_receipts (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    work_item_id uuid NOT NULL,
    run_id uuid,
    decision text NOT NULL CHECK (btrim(decision) <> ''),
    rationale text NOT NULL CHECK (btrim(rationale) <> ''),
    recorded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, work_item_id) REFERENCES studio.work_items (tenant_id, id),
    FOREIGN KEY (tenant_id, work_item_id, run_id) REFERENCES studio.runs (tenant_id, work_item_id, id)
);
CREATE INDEX decision_receipts_by_work ON studio.decision_receipts (tenant_id, work_item_id, recorded_at);

CREATE TABLE studio.pull_requests (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    repository text NOT NULL CHECK (btrim(repository) <> ''),
    number integer NOT NULL CHECK (number > 0),
    state text NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'closed', 'merged')),
    merge_sha text CHECK (merge_sha ~ '^[0-9a-f]{40}$'),
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, repository, number),
    UNIQUE (tenant_id, id, merge_sha),
    CHECK ((state = 'merged') = (merge_sha IS NOT NULL))
);

CREATE TABLE studio.work_pull_requests (
    tenant_id uuid NOT NULL,
    work_item_id uuid NOT NULL,
    pull_request_id uuid NOT NULL,
    PRIMARY KEY (tenant_id, work_item_id, pull_request_id),
    FOREIGN KEY (tenant_id, work_item_id) REFERENCES studio.work_items (tenant_id, id),
    FOREIGN KEY (tenant_id, pull_request_id) REFERENCES studio.pull_requests (tenant_id, id)
);

-- One row is a successful provider report or a separately verified deployment.
-- Only the exact merge can receive evidence. The FK also preserves its identity.
CREATE TABLE studio.deployment_evidence (
    tenant_id uuid NOT NULL,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    pull_request_id uuid NOT NULL,
    merge_sha text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('provider', 'verified')),
    environment text NOT NULL CHECK (btrim(environment) <> ''),
    source_ref text NOT NULL CHECK (btrim(source_ref) <> ''),
    health_passed boolean,
    browser_passed boolean,
    observed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, pull_request_id, merge_sha) REFERENCES studio.pull_requests (tenant_id, id, merge_sha),
    CHECK ((kind = 'verified' AND health_passed IS TRUE AND browser_passed IS TRUE)
        OR (kind = 'provider' AND health_passed IS NULL AND browser_passed IS NULL))
);
CREATE INDEX deployment_evidence_by_pr ON studio.deployment_evidence (tenant_id, pull_request_id, environment, observed_at);

REVOKE ALL ON ALL TABLES IN SCHEMA studio FROM PUBLIC, aiia_studio_app;
DO $$
DECLARE
    table_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY['agents', 'work_items', 'runs', 'decision_receipts',
                                       'pull_requests', 'work_pull_requests', 'deployment_evidence']
    LOOP
        EXECUTE format('ALTER TABLE studio.%I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('ALTER TABLE studio.%I FORCE ROW LEVEL SECURITY', table_name);
        EXECUTE format('CREATE POLICY tenant_access ON studio.%I TO aiia_studio_app
            USING (tenant_id = (SELECT studio.current_tenant_id()))
            WITH CHECK (tenant_id = (SELECT studio.current_tenant_id()))', table_name);
        EXECUTE format('GRANT SELECT, INSERT ON studio.%I TO aiia_studio_app', table_name);
    END LOOP;
END
$$;

GRANT UPDATE (name) ON studio.agents TO aiia_studio_app;
GRANT UPDATE (title, status, blocked_reason, parent_id, parent_kind, agent_id) ON studio.work_items TO aiia_studio_app;
GRANT UPDATE (status) ON studio.runs TO aiia_studio_app;
GRANT UPDATE (state, merge_sha) ON studio.pull_requests TO aiia_studio_app;
GRANT DELETE ON studio.work_pull_requests TO aiia_studio_app;
COMMIT;
