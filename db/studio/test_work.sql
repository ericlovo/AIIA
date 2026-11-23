-- Disposable LOCAL database only. Requires a test superuser to simulate login identities.
-- Apply 001_work.sql first. All test-only identities and records roll back.
\set ON_ERROR_STOP on
BEGIN;
CREATE ROLE studio_test_a LOGIN;
CREATE ROLE studio_test_b LOGIN;
CREATE ROLE studio_test_unmapped LOGIN;
GRANT aiia_studio_app TO studio_test_a, studio_test_b, studio_test_unmapped;
INSERT INTO studio.tenant_principals(role_oid, role_name, tenant_id)
SELECT oid, rolname, '00000000-0000-4000-a000-000000000001'::uuid
FROM pg_roles WHERE rolname = 'studio_test_a';
INSERT INTO studio.tenant_principals(role_oid, role_name, tenant_id)
SELECT oid, rolname, '00000000-0000-4000-a000-000000000002'::uuid
FROM pg_roles WHERE rolname = 'studio_test_b';

SET SESSION AUTHORIZATION studio_test_a;
INSERT INTO studio.work_items(tenant_id,id,kind,title,status)
VALUES (studio.current_tenant_id(),'00000000-0000-4000-b000-000000000001','feature','Synthetic item','blocked');
INSERT INTO studio.pull_requests(tenant_id,id,repository,number,state,merge_sha)
VALUES (studio.current_tenant_id(),'00000000-0000-4000-c000-000000000001','example/demo',1,'merged',repeat('a',40));
INSERT INTO studio.work_pull_requests(tenant_id,work_item_id,pull_request_id)
VALUES (studio.current_tenant_id(),'00000000-0000-4000-b000-000000000001','00000000-0000-4000-c000-000000000001');

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM studio.deployment_evidence WHERE tenant_id = studio.current_tenant_id()) THEN
        RAISE EXCEPTION 'merge unexpectedly generated deployment evidence';
    END IF;
    BEGIN
        INSERT INTO studio.deployment_evidence(tenant_id,pull_request_id,merge_sha,kind,environment,source_ref)
        VALUES (studio.current_tenant_id(),'00000000-0000-4000-c000-000000000001',repeat('b',40),'provider','production','synthetic:check');
        RAISE EXCEPTION 'accepted evidence for a different merge';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
END $$;

INSERT INTO studio.deployment_evidence(tenant_id,pull_request_id,merge_sha,kind,environment,source_ref)
VALUES (studio.current_tenant_id(),'00000000-0000-4000-c000-000000000001',repeat('a',40),'provider','production','synthetic:check');

INSERT INTO studio.agents(tenant_id,id,name)
VALUES (studio.current_tenant_id(),'00000000-0000-4000-d000-000000000001','Synthetic agent');
INSERT INTO studio.runs(tenant_id,id,work_item_id,agent_id,status)
VALUES (studio.current_tenant_id(),'00000000-0000-4000-e000-000000000001','00000000-0000-4000-b000-000000000001','00000000-0000-4000-d000-000000000001','completed');
INSERT INTO studio.decision_receipts(tenant_id,work_item_id,run_id,decision,rationale)
VALUES (studio.current_tenant_id(),'00000000-0000-4000-b000-000000000001','00000000-0000-4000-e000-000000000001','watch','Synthetic review');
DO $$
BEGIN
    BEGIN
        UPDATE studio.decision_receipts SET decision = 'different' WHERE tenant_id = studio.current_tenant_id();
        RAISE EXCEPTION 'application can overwrite receipts';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
    BEGIN
        INSERT INTO studio.deployment_evidence(tenant_id,pull_request_id,merge_sha,kind,environment,source_ref)
        VALUES (studio.current_tenant_id(),'00000000-0000-4000-c000-000000000001',repeat('a',40),'verified','production','synthetic:check');
        RAISE EXCEPTION 'unverified deployment accepted as verified';
    EXCEPTION WHEN check_violation THEN NULL;
    END;
END $$;

RESET SESSION AUTHORIZATION;
SET SESSION AUTHORIZATION studio_test_b;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM studio.work_items WHERE tenant_id = '00000000-0000-4000-a000-000000000001') THEN
        RAISE EXCEPTION 'foreign work is visible';
    END IF;
    BEGIN
        INSERT INTO studio.work_items(tenant_id,kind,title)
        VALUES ('00000000-0000-4000-a000-000000000001','assignment','Synthetic foreign write');
        RAISE EXCEPTION 'foreign write succeeded';
    EXCEPTION WHEN insufficient_privilege THEN NULL;
    END;
    BEGIN
        INSERT INTO studio.work_items(tenant_id,parent_id,parent_kind,kind,title)
        VALUES (studio.current_tenant_id(),'00000000-0000-4000-b000-000000000001','feature','assignment','Synthetic foreign parent');
        RAISE EXCEPTION 'cross-tenant parent succeeded';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
END $$;

RESET SESSION AUTHORIZATION;
SET SESSION AUTHORIZATION studio_test_unmapped;
DO $$
BEGIN
    IF studio.current_tenant_id() IS NOT NULL THEN
        RAISE EXCEPTION 'unmapped role has tenant access';
    END IF;
END $$;
RESET SESSION AUTHORIZATION;
ROLLBACK;
