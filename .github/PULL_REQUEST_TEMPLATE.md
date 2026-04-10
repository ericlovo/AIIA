<!--
Thanks for contributing to AIIA! A few quick checks before you hit submit.
Delete sections that don't apply.
-->

## What

<!-- One or two sentences describing what this PR changes. -->

## Why

<!-- The motivation. What problem does this solve? What did you observe
before this change that prompted you to make it? Link issues if relevant. -->

## How

<!-- High-level approach. Any trade-offs, alternatives you considered, or
places reviewers should look closely. -->

## Testing

<!-- How did you verify this works? Commands run, pytest output, manual
reproduction steps, screenshots for UI changes. -->

```
# paste test output or reproduction steps
```

## Checklist

- [ ] Tests added or updated (or explained why not)
- [ ] `ruff check local_brain/` passes locally
- [ ] `ruff format --check local_brain/` passes locally
- [ ] `pytest --collect-only local_brain/tests/` passes locally
- [ ] `CHANGELOG.md` updated under the `[Unreleased]` / next-version section
- [ ] Docs updated if behavior changed (README, module docstrings)
- [ ] **No proprietary references introduced.** Run this locally before
      pushing and confirm it returns no matches:
      ```
      grep -rni 'cathcap\|realiz\|codeword\|acping\|modivcare\|ck-marketing\|aplora-sales\|aplora-marketing\|family-law-suite\|estate-planner\|content-engine\|aiia-legal\|traction-eos\|ericlovold\|xcai_intelligence\|xcai-intelligence\|josefsberg\|pingorch\|ping_orchestrator\|ping-orch\|retains_defaultapp\|DefaultApp-Direct' --exclude-dir=.git --exclude-dir=node_modules --exclude=CHANGELOG.md --exclude=PULL_REQUEST_TEMPLATE.md --exclude=ci.yml .
      ```
      The CI job runs the same grep — if it finds anything, the build
      fails. If you have a legitimate reason to mention one of these
      names (e.g. in documentation explaining what was removed), add
      that file to the grep's exclude list in `.github/workflows/ci.yml`
      with a comment explaining why.
