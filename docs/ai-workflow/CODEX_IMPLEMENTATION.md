# CODEX_IMPLEMENTATION.md

## Purpose

This file defines the implementation standards Codex must follow while working on the **Urban Operations Intelligence Platform**.

The project is a production-oriented machine learning and data engineering system for NYC 311 service requests. It will include reproducible data ingestion, validation, exploratory analysis, baseline models, advanced models, batch and online predictions, APIs, storage, monitoring, testing, and CI/CD.

The main objective of these rules is to produce code that is:

- Easy for a human engineer to understand.
- Easy to maintain, debug, test, and extend.
- Professionally structured without unnecessary complexity.
- Reusable and modular.
- Safe to modify without breaking existing behavior.
- Suitable for an industry-standard ML and data platform.

These instructions apply to all source code, tests, scripts, notebooks, configuration files, pipelines, APIs, and documentation added or modified by Codex.

---

## 1. Core Engineering Principles

### 1.1 Optimize for human maintainability

Code must be written primarily for future human engineers.

Prefer:

- Clear control flow.
- Explicit names.
- Small focused functions.
- Predictable behavior.
- Simple abstractions.
- Standard library and well-established libraries.
- Easy-to-follow module boundaries.

Avoid:

- Clever one-liners that reduce readability.
- Premature abstraction.
- Deep inheritance hierarchies.
- Hidden side effects.
- Unnecessary metaprogramming.
- Over-engineered design patterns.
- Large functions that perform multiple responsibilities.
- Copy-pasted logic.
- Magic values.
- Broad exception handling that hides failures.

A professional solution is not the most complicated solution. It is the simplest solution that safely satisfies the current requirements and can be extended later.

### 1.2 Preserve existing behavior

Before changing code, Codex must inspect the existing project structure and understand how the relevant functionality currently works.

For every implementation task:

1. Locate the existing modules related to the requested change.
2. Identify reusable functions, services, schemas, constants, models, fixtures, and test utilities.
3. Check current naming and architectural conventions.
4. Reuse existing code when it is correct and suitable.
5. Extend existing abstractions when doing so remains clear.
6. Avoid introducing a parallel implementation for behavior that already exists.
7. Do not delete, rename, move, or rewrite unrelated code.
8. Remove existing code only when it is obsolete, incorrect, unsafe, or directly replaced by the requested implementation.
9. Explain significant removals in the implementation summary.
10. Keep backward compatibility unless the task explicitly requires a breaking change.

The default approach must be:

> Inspect first, reuse second, extend third, create new code only when necessary.

### 1.3 Make scoped changes

Each task should produce the smallest complete change that satisfies the requirement.

Do not:

- Refactor unrelated modules.
- Reformat the entire repository.
- Rename unrelated variables or files.
- Upgrade dependencies unless required.
- Change public interfaces without need.
- Add infrastructure for hypothetical future requirements.
- Mix unrelated cleanup with feature implementation.

When a broader refactor is genuinely required, document:

- Why the existing design prevents a safe implementation.
- What files are affected.
- What behavior must remain unchanged.
- What regression tests protect the change.

---

## 2. Required Workflow Before Coding

### 2.1 Understand the task

Identify:

- The business requirement.
- The expected input.
- The expected output.
- The affected user or system workflow.
- The success criteria.
- Known edge cases.
- Failure behavior.
- Testing requirements.
- Whether the task changes data contracts, API contracts, model behavior, storage, or configuration.

Do not begin implementation based only on a file name or a guessed architecture.

### 2.2 Inspect the repository

Review the relevant:

- Directory structure.
- Existing implementation.
- Shared utilities.
- Configuration system.
- Logging approach.
- Error handling approach.
- Data models and schemas.
- Test structure.
- Dependency management.
- CI configuration.
- Documentation conventions.

Search for similar implementations before creating new ones.

Examples:

- Before creating a date parser, search for an existing date utility.
- Before creating an API client, inspect current ingestion or HTTP client modules.
- Before creating a Pydantic schema, inspect existing schema modules.
- Before creating validation helpers, inspect current data-quality checks.
- Before adding a model pipeline, inspect current preprocessing and model persistence patterns.
- Before adding fixtures, inspect shared test fixtures and factories.

### 2.3 Identify impact

Determine whether the change affects:

- Raw data ingestion.
- Processed datasets.
- Feature definitions.
- Target definitions.
- Train, validation, or test splits.
- Model metrics.
- Saved model artifacts.
- API request or response schemas.
- Database schemas.
- Dashboard outputs.
- Monitoring.
- Existing tests.
- Reproducibility.
- Backward compatibility.

Any change that can alter model training or prediction output must be treated as a behavior change and covered by regression tests.

---

## 3. Project Architecture Rules

### 3.1 Use modular, responsibility-based structure

Organize code by clear responsibility.

The canonical target structure is defined in
[`../architecture/project_structure.md`](../architecture/project_structure.md).
Use that document when choosing locations for source code, configuration, data,
documentation, notebooks, applications, tests, reports, artifacts, and
operational tooling.

The target structure is a guide for incremental development, not a command to
create unused directories or placeholder modules. Codex must:

- Place new files in the canonical location when that area of the platform is
  being implemented.
- Reuse coherent existing structure and migrate it only when the current task
  requires the change.
- Avoid scaffolding future months or components before they are needed.
- Keep tests aligned with the corresponding source responsibility and the test
  categories defined by the canonical structure.
- Document intentional deviations when the canonical location would make the
  implementation less clear or conflict with an established public contract.

### 3.2 Keep module responsibilities clear

Each module should have one primary purpose.

Examples:

- API client modules fetch external data.
- Schema modules define data contracts.
- Validation modules validate data.
- Cleaning modules transform invalid or inconsistent values.
- Feature modules create model inputs.
- Training modules train models.
- Evaluation modules calculate metrics and reports.
- Persistence modules save and load artifacts.
- API routes handle transport concerns.
- Service modules contain application logic.
- Repository modules handle database access.

Do not mix unrelated responsibilities in the same file.

For example, an API route should not download training data, train a model, write arbitrary SQL, calculate business metrics, or format dashboard charts. It should validate the request, call an application service, and return a typed response.

### 3.3 Avoid circular dependencies

Dependency direction should remain clear.

Preferred direction:

```text
API / CLI / Dashboard
        ↓
Application services / Pipelines
        ↓
Domain logic / Features / Models
        ↓
Infrastructure / Storage / External clients
```

Shared utilities must not import higher-level application modules.

---

## 4. Reuse and DRY Rules

### 4.1 Reuse existing components

Before adding code, search for reusable parsers, validators, schemas, constants, enums, feature transformers, metrics, model wrappers, API clients, retry utilities, logging helpers, fixtures, factories, database repositories, and serialization helpers.

Reuse must not be forced when it makes the code harder to understand.

### 4.2 Eliminate meaningful duplication

Do not repeat the same business logic in multiple places.

Extract shared code when:

- The logic is identical or nearly identical.
- The behavior must stay consistent across callers.
- A bug fix would otherwise need to be repeated.
- The abstraction has a clear name and responsibility.

Do not extract code merely because two small blocks look similar. Prefer local clarity over unnecessary abstraction.

### 4.3 Centralize constants and configuration

Do not scatter values such as API base URLs, dataset identifiers, selected agencies, request limits, timeout values, retry counts, random seeds, column names, target thresholds, model artifact paths, database settings, split dates, or metric thresholds.

Use typed settings, environment variables, constants modules, configuration files, and enumerations where appropriate.

Never commit secrets, credentials, tokens, or private connection strings.

---

## 5. Code Readability Standards

### 5.1 Naming

Names must explain intent.

Use names such as:

```python
resolution_duration_hours
is_resolution_target_missed
fetch_service_requests
validate_required_columns
create_time_based_split
```

Avoid vague names such as:

```python
data
temp
obj
result2
process
helper
do_work
```

Short names are acceptable only for conventional local use, such as loop indices or mathematical formulas with clear context.

### 5.2 Function design

Each function must:

- Have one clear responsibility.
- Have a descriptive name.
- Use typed parameters and return types.
- Validate assumptions at appropriate boundaries.
- Return a predictable type.
- Avoid hidden global state.
- Avoid unnecessary mutation.
- Be small enough to understand without excessive scrolling.
- Raise meaningful domain-specific exceptions where appropriate.
- Be independently testable where practical.

Prefer early validation and guard clauses over deeply nested conditionals.

### 5.3 Class design

Use classes when they represent stateful services, external clients, configured pipelines, domain entities, repositories, model wrappers, or components with lifecycle management.

Do not create classes only to group unrelated static functions. Use modules or plain functions when no state or object identity is needed.

Favor composition over inheritance.

### 5.4 Type safety

Use type hints throughout production Python code.

Requirements:

- Type all public functions.
- Type function return values.
- Use precise types instead of `Any` where possible.
- Use `Protocol`, `TypedDict`, dataclasses, or Pydantic models where they improve clarity.
- Avoid unchecked dictionary access for important contracts.
- Keep dataframe schemas documented and validated.
- Use enums or literals for bounded values.

Static type checks should be part of CI when configured in the project.

---

## 6. Comments and Documentation Standards

### 6.1 File-level purpose comment

Every new source file must begin with a concise module docstring explaining the purpose of the file, its responsibility, and important boundaries or assumptions when relevant.

Example:

```python
"""Validate raw NYC 311 records before downstream processing.

This module contains schema, missing-value, uniqueness, and timestamp checks.
It does not modify records; cleaning is handled by the cleaning package.
"""
```

Do not add obvious comments that simply repeat the file name.

### 6.2 Function and class documentation

Every public function and class must have a docstring.

Internal functions must also have docstrings when their purpose, assumptions, inputs, or behavior are not immediately obvious.

A useful function docstring should explain what the function does, important inputs, returned value, important exceptions, side effects, and non-obvious assumptions.

Example:

```python
def create_time_based_split(
    frame: pd.DataFrame,
    train_end: datetime,
    validation_end: datetime,
) -> DatasetSplit:
    """Split records chronologically into train, validation, and test sets.

    Raises:
        MissingColumnError: If the configured timestamp column is unavailable.
        InvalidSplitBoundaryError: If split boundaries are not chronological.
    """
```

### 6.3 Inline comments

Add inline comments only when they explain:

- Why a non-obvious decision exists.
- A business rule.
- A data-quality exception.
- A workaround.
- A performance tradeoff.
- A leakage-prevention rule.
- A model assumption.
- An external API limitation.

Do not comment obvious syntax.

Good example:

```python
# Use created_date rather than closed_date to prevent future information leakage.
records = records.sort_values("created_date")
```

### 6.4 TODO comments

A TODO must include the remaining action, why it is not completed now, and a tracking issue or owner when available.

Avoid vague comments such as `# TODO: improve this`.

Do not leave new TODOs unless the task explicitly permits incomplete work.

---

## 7. Data Engineering Standards

### 7.1 Reproducible ingestion

The ingestion layer should:

- Use the official NYC Open Data API.
- Accept an explicit date range.
- Accept selected agencies or categories.
- Use deterministic query parameters.
- Support pagination.
- Use explicit timeouts.
- Use bounded retry logic for transient failures.
- Log request progress without exposing secrets.
- Persist ingestion metadata.
- Avoid silently returning partial data.
- Validate response structure.
- Support restart or recovery where practical.

Store metadata such as source endpoint, query parameters, extraction timestamp, row count, date range, selected agencies, schema version, and raw file checksum when applicable.

### 7.2 Schema validation

Checks should include:

- Required columns.
- Data types.
- Nullability.
- Timestamp parseability.
- Unique identifiers.
- Allowed categorical values where appropriate.
- Geographic coordinate ranges.
- Duplicate records.
- Chronological consistency.
- Target availability.
- Unexpected schema changes.

A validation failure must produce an actionable error message that identifies the violated rule.

Do not silently drop invalid rows unless the behavior is explicitly defined, logged, measured, and tested.

### 7.3 Data cleaning

Cleaning logic must be deterministic, documented, testable, separated from validation, and applied consistently during training and inference where relevant.

Track the number of rows removed or modified, missing-value treatment, duplicate handling, invalid timestamp handling, category normalization, and outlier handling.

Do not overwrite raw source data.

### 7.4 Leakage prevention

Prevent future information from entering model features.

For each feature, verify whether it would be available at prediction time.

Likely leakage fields include:

- `closed_date`.
- Final status.
- Resolution descriptions produced after closure.
- Resolution action timestamps after complaint creation.
- Derived duration calculated from closure.
- Other post-resolution fields.

Feature code must document prediction-time availability assumptions.

### 7.5 Time-based splits

Use chronological train, validation, and test splits.

Requirements:

- Split boundaries must be explicit and configurable.
- No random shuffling before splitting.
- Fit preprocessing only on the training set.
- Evaluate once on the untouched test set after model selection.
- Test that date ranges do not overlap.
- Test that records are ordered correctly.
- Log split sizes and date ranges.

---

## 8. Machine Learning Standards

### 8.1 Define business metrics first

Before optimizing model metrics, document the operational question, such as staffing demand accuracy, early SLA-risk detection, resolution-time usefulness, complaint-category assistance, or unusual spike detection.

Model metrics must be connected to business decisions.

### 8.2 Always establish baselines

Every modelling task must have a simple baseline before advanced modelling.

Expected baselines include:

- Historical average for volume forecasting.
- Linear regression for resolution time.
- Logistic regression for resolution-risk prediction.
- Majority or rule-based baseline for complaint categorization.
- Simple statistical threshold for anomaly detection.

Advanced models must be compared against the baseline on the same data split and metrics.

### 8.3 Reproducibility

Training code must control random seeds, dataset version, feature version, split boundaries, hyperparameters, library versions, model artifact version, preprocessing artifact, and evaluation configuration.

A saved model must include or reference the exact preprocessing needed for inference.

### 8.4 Separate training and inference

Training code and inference code must share reusable transformation logic but remain operationally separate.

Do not retrain models during API requests.

Inference must load versioned model and preprocessing artifacts, validate request data, return model version metadata, and handle invalid inputs with clear errors.

### 8.5 Evaluation

Use task-appropriate metrics.

For classification, consider precision, recall, F1, ROC-AUC, PR-AUC, calibration error, Brier score, confusion matrix, and threshold-specific operational metrics.

For regression, consider MAE, median absolute error, RMSE where useful, segment-level error, and prediction-interval coverage when implemented.

For forecasting, consider MAE, RMSE, sMAPE or MAPE where mathematically appropriate, error by horizon, and error by agency, category, and geography.

For text classification, consider macro F1, per-class precision and recall, confusion matrix, and low-confidence behavior.

Always include segment-level error analysis for important groups.

### 8.6 Calibration and thresholds

When probabilities drive prioritization:

- Evaluate calibration.
- Compare calibrated and uncalibrated models.
- Choose decision thresholds using business costs.
- Store the selected threshold in configuration.
- Test threshold behavior.
- Avoid hard-coding `0.5` without justification.

### 8.7 Explainability

Explainability code must distinguish global model behavior, local explanation, association, and causality.

Feature importance and SHAP values must not be described as proof of causation.

---

## 9. API and Service Standards

### 9.1 Typed contracts

FastAPI request and response bodies must use typed Pydantic schemas.

Requirements:

- Validate required fields.
- Define field descriptions.
- Constrain values where appropriate.
- Use stable response shapes.
- Version public APIs when breaking changes are possible.
- Return meaningful HTTP status codes.
- Do not expose internal stack traces.
- Include model version in prediction responses.

### 9.2 Thin routes

API route handlers should validate transport-level input, resolve dependencies, call an application service, map known exceptions, and return a typed response.

Business logic belongs in services, not route handlers.

### 9.3 Error handling

Use specific exception types such as `DataValidationError`, `SchemaMismatchError`, `ModelArtifactNotFoundError`, `PredictionInputError`, `ExternalApiError`, and `ConfigurationError`.

Do not use `except Exception: pass`.

Catch broad exceptions only at application boundaries where they are logged and converted into a safe response.

### 9.4 Logging

Use structured logging with relevant context such as pipeline name, run ID, dataset version, model version, request correlation ID, record counts, processing duration, and error category.

Do not log secrets, access tokens, full connection strings, unnecessary personal data, or entire raw complaint descriptions by default.

Do not use `print()` for production observability.

---

## 10. Database and Persistence Standards

### 10.1 Repository boundary

Database access should be isolated behind repository or storage modules.

Do not scatter SQL across API routes, notebooks, and model code.

Repositories should use parameterized queries or an ORM safely, define transaction boundaries, return typed objects, avoid leaking database details into business logic, handle expected errors, and be integration-tested.

### 10.2 Schema changes

Database schema changes must use migrations.

A migration must be reviewable, reversible where practical, avoid destructive changes without explicit approval, preserve existing data when possible, and include migration testing for critical tables.

### 10.3 Model artifacts

Saved artifacts must be versioned and should track model identifier, model version, training run ID, dataset version, feature version, preprocessing version, training timestamp, evaluation metrics, and code commit SHA when available.

Never overwrite a production artifact without creating a new version.

---

## 11. Testing Requirements

Every feature or bug fix must include appropriate automated tests.

### 11.1 Unit tests

Unit tests should cover parsing, validation, cleaning, feature engineering, split logic, metric calculations, threshold behavior, mapping, serialization, service-level decisions, and error handling.

Unit tests must be fast, deterministic, independent, clear, and free from real network calls and production databases.

### 11.2 Integration tests

Integration tests should cover API clients with mocked responses, pipeline stages working together, database repositories against a test database, model save/load behavior, preprocessing plus inference, FastAPI service dependencies, and MLflow integration when introduced.

Use realistic but small test data.

### 11.3 API contract tests

API contract tests must verify request validation, response schema, HTTP status codes, error shape, model version fields, backward-compatible behavior, invalid inputs, missing inputs, and boundary values.

### 11.4 Data-quality tests

Data-quality tests should cover:

- Required columns.
- Duplicate `unique_key` values.
- Missing critical timestamps.
- Invalid timestamps.
- Negative resolution durations.
- Impossible latitude and longitude values.
- Unexpected category changes.
- Empty extracts.
- Schema drift.
- Leakage columns in feature sets.

### 11.5 Regression tests

Regression testing is mandatory.

Add regression tests when fixing a bug or changing feature engineering, target logic, cleaning, split logic, preprocessing, model serialization, API behavior, threshold logic, database mapping, or other important behavior.

For a bug fix:

1. Write a test that reproduces the bug.
2. Confirm the test fails before the fix.
3. Implement the fix.
4. Confirm the test passes.
5. Run related existing tests.

Regression tests should protect observable behavior, not implementation details.

For deterministic baselines, use small golden datasets and expected outputs where practical.

For non-deterministic models:

- Fix seeds.
- Assert acceptable ranges or invariants.
- Avoid fragile exact floating-point equality.
- Verify interfaces, shapes, schemas, and minimum quality thresholds.

### 11.6 End-to-end tests

Add end-to-end tests for critical workflows such as:

```text
raw API response
→ ingestion
→ validation
→ cleaning
→ feature generation
→ model inference
→ prediction storage
→ API response
```

Use a small controlled dataset and isolated test infrastructure.

### 11.7 Performance tests

Where relevant, test API latency, batch throughput, memory usage, large-page ingestion, database query performance, and model loading time.

Do not make performance claims without measurement.

### 11.8 Test naming

Test names should describe behavior.

Prefer:

```python
def test_time_split_places_future_records_in_test_set() -> None:
    ...
```

Use Arrange–Act–Assert structure where it improves clarity.

---

## 12. Test Execution Rules

Before considering a task complete, Codex must run:

1. Tests for the directly changed module.
2. Related unit tests.
3. Related integration or contract tests.
4. Relevant regression tests.
5. Static analysis and formatting checks.
6. The broader test suite when feasible.

Do not claim tests passed unless they were actually executed.

If a test cannot run, report the exact command attempted, why it could not run, whether the failure is related to the change, and what remains unverified.

Never hide failing tests.

---

## 13. Notebooks and Exploratory Analysis

Notebooks are allowed for exploration and communication, but production logic must not live only in notebooks.

Notebooks should:

- Have a clear title and objective.
- Use ordered sections.
- Avoid hidden state.
- Run top to bottom.
- Use reusable functions from `src/`.
- Use fixed random seeds.
- Avoid hard-coded local paths.
- Keep outputs reasonably sized.
- Document important findings.
- Separate exploration from production pipelines.

Any logic required by training, evaluation, or production must be moved into tested Python modules.

---

## 14. Dependency Management

Before adding a dependency:

1. Check whether the existing stack already solves the problem.
2. Prefer actively maintained, widely used packages.
3. Confirm license suitability.
4. Add a pinned or constrained version according to repository standards.
5. Document why the dependency is needed.
6. Avoid overlapping libraries for the same purpose.
7. Consider security and compatibility.

Do not upgrade unrelated dependencies during a feature task.

---

## 15. Security and Privacy

Codex must:

- Keep secrets in environment variables or secret managers.
- Never commit credentials.
- Validate external input.
- Use parameterized database queries.
- Restrict file-system paths.
- Avoid unsafe deserialization.
- Treat model artifact loading as a trusted-source operation.
- Sanitize user-visible errors.
- Minimize storage of raw complaint text when unnecessary.
- Avoid logging sensitive location information without need.
- Use least-privilege access.

Do not load untrusted pickle or joblib artifacts.

---

## 16. Configuration and Environment Rules

Use explicit environment-based configuration for development, testing, staging, and production.

Configuration must be typed, fail fast when required settings are missing, avoid environment-specific logic scattered through the code, be testable, and never contain committed secrets.

Provide `.env.example` entries for new environment variables without real values.

---

## 17. CI/CD Quality Gates

GitHub Actions should eventually enforce:

- Dependency installation.
- Formatting checks.
- Linting.
- Static type checking.
- Unit tests.
- Integration tests where practical.
- API contract tests.
- Data-quality tests.
- Security or dependency scanning.
- Build validation.
- Docker image build.
- Coverage threshold if adopted.

A pull request must not be considered complete when required CI checks are failing.

Do not weaken quality gates merely to make a change pass.

---

## 18. Backward Compatibility

Before changing an existing interface, check all callers.

Interfaces include function signatures, API schemas, database schemas, configuration keys, CLI commands, model input schemas, feature names, artifact formats, and dashboard contracts.

When a breaking change is unavoidable:

- Document it.
- Add migration logic where practical.
- Version the interface.
- Update all callers.
- Add tests for the new contract.
- Remove old behavior only when explicitly approved.

---

## 19. Refactoring Rules

Refactor when it materially improves correctness, testability, reuse, readability, maintainability, performance, or security.

Before refactoring:

- Add or identify tests protecting current behavior.
- Define the intended unchanged behavior.
- Keep the refactor separate from unrelated feature changes when possible.

After refactoring:

- Run regression tests.
- Confirm public behavior remains unchanged.
- Report any intentionally changed behavior.

---

## 20. Definition of Done

A task is complete only when all applicable items are satisfied.

### Implementation

- The requested behavior is implemented.
- The implementation follows the current project architecture.
- Existing reusable code was used where appropriate.
- No unnecessary code was removed.
- No meaningful duplication was introduced.
- Public functions and classes are documented.
- New files include purpose docstrings.
- Types are added.
- Errors are handled explicitly.
- Logging is appropriate.
- Configuration is not hard-coded.
- Security and privacy concerns were considered.

### Testing

- Unit tests were added or updated.
- Regression tests were added for bug fixes and behavior changes.
- Integration tests were added when boundaries changed.
- API contract tests were added when API behavior changed.
- Data-quality tests were added when data rules changed.
- Tests were actually executed.
- Existing relevant tests still pass.
- No failing test was hidden or ignored.

### Documentation

- Relevant README or technical documentation was updated.
- New configuration values were documented.
- New commands were documented.
- Data assumptions were documented.
- Model or feature changes were documented.
- Breaking changes were documented.

### Reviewability

- The change is focused.
- File names and symbols are clear.
- Complex decisions are explained.
- The final summary identifies what changed.
- The final summary identifies tests run.
- The final summary identifies risks or remaining limitations.

---

## 21. Required Codex Completion Report

At the end of every implementation task, Codex must provide:

```markdown
## Implementation Summary

### What changed
- ...

### Reused existing components
- ...

### Files added
- ...

### Files modified
- ...

### Tests added or updated
- ...

### Validation performed
- `command`
- Result: passed / failed / not run

### Compatibility
- Existing behavior preserved:
- Breaking changes:

### Risks or limitations
- ...

### Follow-up work
- None, or explicitly listed items.
```

Do not state that the task is complete when critical validation remains unperformed.

---

## 22. Prohibited Practices

Codex must not:

- Rewrite unrelated code.
- Remove working behavior without justification.
- Duplicate existing logic.
- Add broad abstractions without current use.
- Hard-code secrets or environment-specific paths.
- Use `print()` as production logging.
- Swallow exceptions.
- Use random train/test splitting for time-dependent production evaluation.
- Fit preprocessing on validation or test data.
- Use future fields as prediction features.
- Train models inside API requests.
- Commit large generated datasets or model binaries unless repository policy allows it.
- Put core production logic only in notebooks.
- Claim a test passed without running it.
- Disable tests to make CI green.
- Lower quality thresholds without explanation.
- Introduce a breaking contract silently.
- Add untracked TODOs as a substitute for completing work.
- Describe feature importance as causation.
- Modify data silently without measuring and reporting the change.

---

## 23. Decision Priorities

When implementation choices conflict, use this priority order:

1. Correctness.
2. Data leakage prevention.
3. Safety and security.
4. Reproducibility.
5. Backward compatibility.
6. Testability.
7. Human readability.
8. Maintainability.
9. Reuse.
10. Performance based on measured need.
11. Extensibility based on known requirements.
12. Elegance.

---

## 24. Project-Specific Initial Focus

For Month 1 of the Urban Operations Intelligence Platform, Codex should prioritize:

1. Business metric definitions.
2. Reproducible NYC 311 API ingestion.
3. Raw-data schema validation.
4. Missing-value analysis.
5. Duplicate and inconsistency detection.
6. Explicit target-definition logic.
7. Leakage analysis.
8. Time-based train, validation, and test splits.
9. Historical-average baseline.
10. Linear or logistic regression baseline.
11. Rule-based complaint-category baseline.
12. Initial exploratory analysis.
13. Automated unit, regression, and data-quality tests.
14. Reproducible commands and documentation.

Advanced modelling and production infrastructure should not be introduced prematurely unless required by the current task.

---

## Final Instruction to Codex

Before every change:

> Understand the requirement, inspect the existing implementation, reuse what is already correct, make the smallest safe change, document non-obvious decisions, and protect the behavior with automated regression tests.

The result must remain understandable and maintainable by a professional human engineering team.
