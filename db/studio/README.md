# AIIA Studio work records

This schema is for the hosted Studio backend. Each tenant can connect a feature to an assignment, its agent, run and review decision, then follow a linked pull request to deployment evidence. A merge can exist without a deployment. Provider success and operator verification are separate evidence types.

The Mini still owns its registries, run ledger, BD packets and memory. This migration creates empty tables. It does not install a runtime adapter, import local records, start a worker, or require the Mini to connect to Neon.

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
