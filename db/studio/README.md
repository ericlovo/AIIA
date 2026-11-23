# AIIA Studio work records

This schema is for the hosted Studio backend. Each tenant can connect a feature to an assignment, its agent, run and review decision, then follow a linked pull request to deployment evidence. A merge can exist without a deployment. Provider success and operator verification are separate evidence types.

The Mini still owns its registries, run ledger, BD packets and memory. This migration creates empty tables. It does not install a runtime adapter, import local records, start a worker, or require the Mini to connect to Neon.

## Connect Codex to a development project

The Neon and Neon Postgres skills live in `.agents/skills` with their source
recorded in `skills-lock.json`. From a trusted checkout with the Neon CLI
installed, select the project and branch and register its OAuth MCP endpoint:

```sh
neon link --project-id "$STUDIO_NEON_PROJECT_ID" --branch dev --no-env-pull --no-config -y
neon mcp --oauth --project --project-id "$STUDIO_NEON_PROJECT_ID" --agent codex -y
```

Set `STUDIO_NEON_PROJECT_ID` to your project's ID first. The CLI writes local
branch context to `.neon` and the project-specific endpoint to
`.codex/config.toml`. Open a fresh Codex session, authenticate the Neon MCP
server when prompted, and select the development branch explicitly when
calling `run_sql`. This catalog query checks that the migration is present.
Expect eight tables and seven with forced row-level security. The eighth table
maps database logins to tenants and is accessible only to the migration owner.

```sql
SELECT count(*) AS studio_tables,
       count(*) FILTER (WHERE relrowsecurity AND relforcerowsecurity) AS forced_rls
FROM pg_catalog.pg_class
WHERE relnamespace = 'studio'::regnamespace AND relkind = 'r';
```

## Apply on a new development branch

Use the migration owner on the dedicated AIIA Studio project. Select the project and development branch in Neon, then configure a local libpq service named `aiia_studio_dev` with its direct endpoint, role, database and TLS settings. Keep the password in a secret store. Run from the repository root:

```sh
PGSERVICE=aiia_studio_dev psql -X -v ON_ERROR_STOP=1 -f db/studio/001_work.sql
```

The migration runs in one transaction. Apply it once to an empty target. It does not use the Brain's `DATABASE_URL`.

## Tenant access

The hosted backend must authenticate the caller, authorize tenant membership and use that tenant's database login. The migration owner is for maintenance only. Each application login must be a non-owner without superuser, role administration or BYPASSRLS privileges. Grant membership in `aiia_studio_app` and, as the owner, insert the login's OID, name and authorized tenant UUID into `studio.tenant_principals`. Rebind deliberately if a login is recreated. An unmapped login has no tenant access. Use the tenant UUID explicitly on every read and write.

Run this from the backend using the authorized tenant's connection and a bound UUID. It shows whether the PR for a work item has merged and whether that particular merge has evidence for the requested environment.

```sql
SELECT w.title, w.status, p.repository, p.number, p.state, p.merge_sha,
       EXISTS (
           SELECT 1 FROM studio.deployment_evidence e
           WHERE e.tenant_id = w.tenant_id AND e.pull_request_id = p.id
             AND e.merge_sha = p.merge_sha AND e.environment = $2 AND e.kind = 'verified'
       ) AS verified_deployed,
       EXISTS (
           SELECT 1 FROM studio.deployment_evidence e
           WHERE e.tenant_id = w.tenant_id AND e.pull_request_id = p.id
             AND e.merge_sha = p.merge_sha AND e.environment = $2 AND e.kind = 'provider'
       ) AS provider_reported
FROM studio.work_items w
LEFT JOIN studio.work_pull_requests l ON (l.tenant_id, l.work_item_id) = (w.tenant_id, w.id)
LEFT JOIN studio.pull_requests p ON (p.tenant_id, p.id) = (l.tenant_id, l.pull_request_id)
WHERE w.tenant_id = $1;
```

## Validate locally

On a disposable local PostgreSQL database, apply `001_work.sql`, then run
`psql -X -v ON_ERROR_STOP=1 -f db/studio/test_work.sql` as its local
superuser. The test switches between synthetic tenant logins and rolls back
its fixtures. Never grant the hosted application superuser privileges.

## Review a synthetic BD packet on Neon dev

From a development workstation, apply `002_bd_packets.sql` once to the same
dev branch as `001_work.sql`, using the migration owner. The migration adds
replay packets and links each packet decision to its work item and run.
Use a separate synthetic tenant login provisioned as described above for the
preview. Set `STUDIO_DESK_DEV_DSN` to that login's TLS connection string
and `STUDIO_DESK_DEV_TENANT_ID` to its mapped UUID in the server environment.

Install the `studio-postgres` Python extra in the development environment
and run `npm ci && npm run build` in `dashboard/`. From the repository root,
seed the fixture and start the loopback preview:

```sh
python -c 'import os; from local_brain.command_center.bd_desk_postgres import PostgresDesk; print(PostgresDesk(os.environ["STUDIO_DESK_DEV_DSN"], os.environ["STUDIO_DESK_DEV_TENANT_ID"]).seed())'
python -m local_brain.command_center.bd_desk_dev --port 8765
```

Open `http://127.0.0.1:8765/#/desk`, select **Watch**, follow **View receipt
in Record**, then **Inspect run**. The run ID should match the seed command's
output. Seeding again preserves the existing decision.

To exercise those steps in a browser, run the following from the repository
root with the preview running. Set `DESK_NEON_RUN_ID` to the printed UUID.
On later runs the test finds the existing decision in Record and reloads it.

```sh
DESK_NEON_STUDIO_URL=http://127.0.0.1:8765 node dashboard/tests/desk-neon-dev.browser.mjs
```
