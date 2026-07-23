# CURSOR_RULES.md

## Purpose

This file defines how Cursor must be used in the **Urban Operations Intelligence Platform** under the agreed AI-assisted engineering workflow.

Cursor is the local development, inspection, polishing, and small-edit environment used after Codex completes the main implementation.

The agreed workflow is:

```text
ChatGPT
→ planning, architecture, concept explanation, acceptance criteria,
  and implementation-prompt preparation

Codex
→ complete end-to-end implementation, automated tests,
  regression coverage, validation, and technical documentation

Cursor
→ local execution, inspection, small edits, documentation polishing,
  minor corrections, environment-specific debugging, and Git management

Claude
→ final independent review
```

Cursor is important, but it is not the primary implementation agent in this workflow. Cursor should not silently become a second full implementer after Codex finishes.

---

## 1. Cursor Role and Responsibilities

Cursor acts as the developer's local control centre after Codex implementation.

Cursor is responsible for:

- Opening and navigating the repository.
- Inspecting Codex-created or modified files.
- Reviewing the Git diff.
- Running notebooks, scripts, tests, services, and commands locally.
- Inspecting dataframe outputs, reports, charts, logs, and API responses.
- Performing small, focused, low-risk edits.
- Improving documentation, comments, Markdown, and notebook presentation.
- Resolving local environment, path, interpreter, or tooling issues.
- Confirming that the implementation works in the developer's environment.
- Removing temporary debugging output before review.
- Preparing the final working tree for Claude.
- Committing and pushing only after verification.

Cursor must help the developer understand and verify the implementation rather than blindly accepting generated code.

---

## 2. Scope of Allowed Cursor Work

### 2.1 Local execution and inspection

Cursor may be used to:

- Run Jupyter notebooks from top to bottom.
- Inspect individual notebook cells and outputs.
- Start FastAPI or dashboard applications locally.
- Run data-ingestion commands against approved development inputs.
- Run unit, regression, integration, contract, data-quality, and end-to-end tests.
- Run formatting, linting, type checks, packaging, and Docker commands.
- Inspect logs and stack traces.
- Compare generated reports and artefacts.
- Review Git status, diff, history, and changed files.
- Confirm project-relative paths work locally.
- Verify environment variables through safe example configuration.

Execution and inspection do not authorize unrelated code changes.

### 2.2 Small code edits

Cursor may make small code edits when all of the following are true:

- The required behaviour is already understood.
- The change is isolated.
- The change does not redesign architecture.
- The change does not alter a public contract unexpectedly.
- The change does not affect multiple subsystems.
- The developer can review the complete edit directly.
- Appropriate tests can be run immediately.

Examples include:

- Fixing a typo in a variable, message, or label.
- Correcting a project-relative path.
- Fixing a small import problem.
- Correcting a clearly isolated null check.
- Adjusting a non-breaking configuration value.
- Fixing a small chart label or notebook display issue.
- Removing a temporary debug statement.
- Applying a tiny, confirmed Claude Low-severity finding.
- Correcting an obvious documentation-code mismatch without changing behaviour.

### 2.3 Documentation work

Cursor is the preferred tool for small documentation and presentation updates, including:

- README corrections.
- Markdown wording.
- Docstring clarity.
- Comment improvements.
- Notebook headings and explanatory cells.
- Command examples.
- `.env.example` descriptions without secrets.
- Changelog or implementation-summary corrections.
- Report formatting.

Documentation changes must remain truthful and consistent with actual behaviour.

### 2.4 Notebook polishing

After Codex implements a notebook, Cursor may:

- Run every cell in order.
- Inspect warnings and errors.
- Reduce excessive output.
- Improve headings and Markdown explanations.
- Correct chart titles, axis labels, legends, and table presentation.
- Confirm project-relative paths.
- Confirm findings match actual outputs.
- Fix a small isolated notebook issue.

Cursor must not silently change the business interpretation, target rule, exclusion policy, leakage decision, or modelling conclusion merely to make a notebook look cleaner.

---

## 3. Changes That Must Return to Codex

A change must normally return to Codex when it involves any of the following:

- A new feature or complete implementation ticket.
- Multiple connected production modules.
- Architecture or module-boundary changes.
- Reusable ingestion, validation, cleaning, or feature pipelines.
- Target-definition logic.
- Data leakage or prediction-time availability.
- Time-based split logic.
- Model training, preprocessing, calibration, or persistence.
- API request or response contracts.
- Database schemas, repositories, or migrations.
- Docker, CI/CD, MLflow, monitoring, or deployment behaviour.
- Security or privacy behaviour.
- Backward compatibility.
- A bug requiring substantial regression coverage.
- A Claude Blocker or High finding.
- A Medium finding that affects several files or important behaviour.
- A fix that expands beyond a small, directly reviewable edit.

Cursor must not continue expanding a small edit until it becomes an unplanned implementation project.

When the boundary is crossed:

1. Stop the broad edit.
2. Record the observed problem and evidence.
3. Preserve useful logs or failing test output.
4. Ask ChatGPT to prepare a focused Codex correction prompt.
5. Return the issue to Codex.

---

## 4. Repository Inspection Before Editing

Before any Cursor edit:

1. Read the current task and Codex implementation summary.
2. Inspect the relevant file and surrounding modules.
3. Search for existing utilities, constants, schemas, and tests.
4. Check whether the change affects callers or contracts.
5. Confirm that the edit belongs within Cursor's small-change boundary.
6. Identify the targeted validation command.

Do not edit a file only because it appears in the diff. Understand why it changed.

---

## 5. Human Maintainability Standards

Cursor edits must preserve the same professional standards required from Codex.

Code must remain:

- Easy to understand.
- Easy to maintain.
- Easy to debug.
- Easy to test.
- Modular and reusable where appropriate.
- Explicit rather than clever.
- Consistent with the current project structure.

Cursor must:

- Use descriptive names.
- Preserve focused function responsibilities.
- Add or maintain accurate type hints.
- Keep module docstrings accurate.
- Keep public function and class docstrings accurate.
- Add comments only where they explain a non-obvious reason.
- Preserve specific error handling.
- Preserve structured logging.
- Avoid hidden side effects.
- Avoid copy-pasted business logic.
- Avoid unnecessary abstractions.

A small edit is not exempt from quality standards.

---

## 6. Reuse and Change Safety

Before creating a helper or adding logic in Cursor, search for an existing implementation.

Cursor must not:

- Create a parallel utility that duplicates existing behaviour.
- Copy business logic into a notebook or second module.
- Rename public interfaces casually.
- Remove working code without justification.
- Reformat unrelated files.
- Upgrade unrelated dependencies.
- move files without need.
- Add future-stage scaffolding.
- change configuration keys without checking callers.
- modify raw data files manually.

Default rule:

> Inspect first, reuse second, make the smallest safe edit, and escalate broader work to Codex.

---

## 7. Comments and Documentation Rules

### 7.1 Module documentation

Do not remove or weaken module-purpose docstrings.

When a small Cursor edit changes a file's responsibility or assumptions, update the module docstring accurately. If the responsibility changes substantially, the work likely belongs in Codex.

### 7.2 Function and class documentation

Public functions and classes must retain useful docstrings describing relevant inputs, outputs, exceptions, side effects, and assumptions.

Do not add comments that merely repeat syntax.

Useful comments explain:

- Business rules.
- Data-quality policy.
- Leakage prevention.
- External API limitations.
- Compatibility workarounds.
- Non-obvious performance decisions.

### 7.3 TODO rules

Do not add vague TODOs.

A TODO must explain:

- What remains.
- Why it is deferred.
- Who or which issue owns it when available.

Cursor must not use a TODO to hide incomplete Codex implementation. Return missing core work to Codex.

---

## 8. Testing Rules for Cursor Edits

Every behavioural edit made in Cursor must be validated.

### 8.1 Documentation-only changes

For true documentation-only changes:

- Verify links and paths where practical.
- Confirm commands match the repository.
- Ensure statements match actual behaviour.
- Run Markdown or documentation checks if configured.

### 8.2 Code changes

For code changes:

- Run the directly affected test.
- Run related unit tests.
- Run relevant regression tests.
- Run linting and type checks when applicable.
- Run integration or contract tests if the small edit crosses a boundary.

### 8.3 Bug fixes

For a bug fix:

1. Reproduce the issue.
2. Add or confirm a regression test when practical.
3. Apply the smallest safe correction.
4. Confirm the regression test passes.
5. Run related tests.

If meaningful regression protection requires a broader implementation, return the issue to Codex.

### 8.4 Honest reporting

Do not claim a test passed unless it was run.

If a test cannot run, record:

- The command.
- The observed error.
- Whether the failure is environment-related or code-related.
- What remains unverified.

Never disable or weaken tests merely to prepare the branch for Claude.

---

## 9. Data and ML Safety

Small Cursor edits must not compromise data or ML correctness.

Cursor must not casually change:

- Target formulas.
- Eligible-population rules.
- Exclusion policies.
- Timestamp assumptions.
- Leakage exclusions.
- Split boundaries or ordering.
- Preprocessing fit behaviour.
- Feature availability assumptions.
- Model thresholds.
- Calibration logic.
- Evaluation metrics.
- Artefact compatibility.

Any such change should normally return to ChatGPT for requirement confirmation and then Codex for implementation and regression testing.

During local inspection, Cursor should specifically verify:

- Notebooks use the intended data.
- Dates parse consistently.
- Train, validation, and test periods do not overlap.
- Preprocessing is not fitted on validation or test data.
- Future fields are not used as features.
- Reported metrics match generated outputs.
- Model and preprocessing artefacts load together.

---

## 10. API, Database, Security, and Configuration Safety

Cursor must not make casual changes to:

- Public API schemas.
- HTTP status-code behaviour.
- Database schema or migrations.
- Transaction boundaries.
- Authentication or authorization.
- Secret handling.
- SQL construction.
- Model artefact loading.
- Production environment configuration.

Small safe edits may include correcting a documented environment variable name or local development path when all callers and tests are checked.

Never:

- Commit credentials.
- Add secrets to documentation.
- expose stack traces.
- Log tokens, passwords, or full connection strings.
- Load untrusted pickle or joblib artefacts.
- Replace safe parameterized queries with string construction.

Substantial changes in these areas return to Codex.

---

## 11. Local Verification Workflow

After Codex completes a task, use Cursor in this order:

1. Read the Codex implementation summary.
2. Inspect `git status` and the complete diff.
3. Confirm only expected files changed.
4. Open key production modules and tests.
5. Run targeted tests supplied by Codex.
6. Run notebooks or services locally where relevant.
7. Inspect generated tables, charts, reports, API responses, and logs.
8. Make only permitted small edits.
9. Rerun affected tests after every behavioural edit.
10. Remove temporary debug output.
11. Prepare a final local verification summary.
12. Send the final working state to Claude.

---

## 12. Git and Diff Hygiene

Before Claude review:

- Keep the diff focused.
- Remove temporary files.
- Remove local data outputs that should not be committed.
- Do not commit secrets or `.env` values.
- Do not commit large model artefacts or datasets unless repository policy allows it.
- Avoid unrelated formatting changes.
- Ensure generated reports intended for version control are reproducible.
- Confirm ignored files remain ignored.
- Review every modified file.

Cursor should not hide Codex changes inside a large unreviewable manual diff.

---

## 13. Preparing the Claude Review Package

The review package should contain or identify:

- The original ChatGPT task and acceptance criteria.
- The final Git diff.
- The Codex implementation summary.
- A list of Cursor edits.
- Commands run locally.
- Test results.
- Notebook or runtime verification performed.
- Known limitations.
- Any unverified areas.

Claude reviews the final state, including Cursor edits.

---

## 14. Handling Claude Findings

Route findings as follows:

### Return to Codex

- Blocker findings.
- High findings.
- Architecture problems.
- Data leakage.
- Target or split errors.
- Missing important regression coverage.
- Multi-file logic defects.
- API or database contract problems.
- Security issues.
- Model or preprocessing incompatibility.
- CI/CD or production failures.

### Handle in Cursor

- Typographical errors.
- Small wording corrections.
- Markdown formatting.
- Minor comment or docstring clarity.
- Clearly isolated low-risk edits.
- Low findings that require no architecture or behavioural redesign.

### Return to ChatGPT

- Ambiguous requirements.
- Disputed business rules.
- New scope decisions.
- Architecture choices not covered by the original prompt.

Do not apply Claude suggestions blindly. Verify each finding against the current implementation.

---

## 15. Cursor Completion Summary

Before sending the branch to Claude, provide:

```markdown
## Cursor Local Verification Summary

### Codex implementation inspected
- Yes / No

### Small edits made in Cursor
- `path/to/file`: description

### Documentation edits
- ...

### Commands run
- `command`
  - Result: passed / failed / not run

### Notebook or runtime verification
- ...

### Git diff check
- Unrelated files changed: yes / no
- Temporary debugging code removed: yes / no
- Secrets or local-only files detected: yes / no

### Issues returned to Codex
- None, or list the issue and evidence.

### Remaining unverified areas
- None, or list them explicitly.
```

---

## 16. Prohibited Cursor Practices

Cursor must not:

- Rebuild the complete feature after Codex without a new approved plan.
- Expand a small edit into an undocumented refactor.
- Change core business logic casually.
- Modify target, feature, or split logic without full review and testing.
- Introduce duplicate implementations.
- Remove tests to make validation pass.
- Claim unexecuted tests passed.
- Hide Codex failures through manual workarounds.
- Commit machine-specific absolute paths.
- Commit secrets.
- Edit raw source data manually.
- Make silent API or database contract changes.
- Replace production logging with `print()`.
- swallow exceptions.
- change model claims without evidence.
- describe feature importance as causation.
- approve its own work as the final independent reviewer.

---

## 17. Definition of Done for Cursor Stage

The Cursor stage is complete when:

- The Codex implementation has been inspected.
- The relevant code runs locally where practical.
- Required notebooks run top to bottom where applicable.
- Targeted tests pass or failures are documented honestly.
- Small edits are focused and tested.
- Documentation matches actual behaviour.
- The diff contains no unrelated changes.
- Temporary debug code is removed.
- Secrets and local artefacts are not included.
- Substantial issues have been returned to Codex.
- The final review package is ready for Claude.

---

## Final Instruction to Cursor

> Use Cursor to inspect, run, understand, polish, and make small safe corrections to the complete Codex implementation. Do not silently take over unfinished architecture or core feature work; return substantial changes to Codex and prepare a clean, verified final state for Claude.
