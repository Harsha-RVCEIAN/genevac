---
trigger: always_on
---

# Antigravity Agent Workflow

## Required operating loop

1. Identify the currently approved phase/task from `00_EXECUTION_ORDER.md`.
2. Inspect repository status, relevant source, tests, and documentation.
3. Map requirements to existing implementation before proposing new code.
4. Separate verified facts from assumptions and unknowns.
5. Present a concise plan and exact intended file list.
6. Stop for approval when the plan or phase requires it.
7. Implement only the approved scope.
8. Run relevant tests and report actual results.
9. Summarize changed files, behavior, evidence, limitations, and next decision.

## Phase gates

The execution plan requires approval after every phase. Do not automatically continue to the next numbered document. A request to inspect is not permission to edit; a request to document is not permission to change source code.

## Required status report

```text
Phase:
Status: NOT STARTED / IN PROGRESS / BLOCKED / COMPLETE
Files and symbols inspected:
Existing implementation reused:
Proposed files to change:
Files actually changed:
Commands actually executed:
Tests/experiments actually run:
Actual results:
Assumptions and unresolved questions:
Risks/limitations:
Approval needed:
```

## When blocked

Ask one focused question with the exact missing decision or evidence. Do not guess, silently substitute an algorithm, or proceed with a potentially incompatible interface.
