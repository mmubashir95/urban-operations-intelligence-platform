# AI Tool Rules Review and Preservation Report

## Agreed workflow

```text
ChatGPT → planning and implementation-prompt preparation
Codex   → complete end-to-end implementation
Cursor  → local inspection, small edits, documentation, and minor fixes
Claude  → final independent review
```

## Preservation result

| File | Original lines | Updated lines | Original lines preserved in order |
|---|---:|---:|---|
| CODEX_IMPLEMENTATION | 1034 | 1192 | Yes |
| CLAUDE_REVIEW | 1275 | 1417 | Yes |

The original Codex and Claude rule content was not shortened. New workflow sections were inserted after each file's Purpose section. Existing implementation, architecture, testing, data, ML, API, database, security, CI/CD, compatibility, completion, and review rules remain present in their original order.

## Codex changes

Added a workflow section that establishes Codex as the primary end-to-end implementation engineer receiving the ChatGPT-prepared prompt. It defines prompt readiness, implementation ownership, Cursor handoff, Claude handoff, handling of review findings, and the complete workflow gate.

## Claude changes

Added a workflow section that establishes Claude as the final independent reviewer after Codex implementation and Cursor local verification. It adds a first-pass no-edit rule, final-branch review expectations, finding ownership, re-review process, and workflow approval gate.

## Cursor file

Created a dedicated Cursor rule file covering:

- Local execution and inspection.
- Small code edits.
- Documentation and notebook polishing.
- Boundaries requiring return to Codex.
- Testing and regression safety.
- Data and ML safety.
- API, database, security, and configuration safeguards.
- Git and diff hygiene.
- Claude review preparation.
- Finding routing and definition of done.
