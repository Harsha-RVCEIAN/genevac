---
trigger: always_on
---

# Git and Workspace Safety Rules

## Before work

Inspect and report:
- `git status --short --branch`
- current branch and recent commits;
- `git stash list`;
- relevant untracked and modified files.

Preserve all pre-existing user changes, untracked files, and stashes.

## Forbidden without explicit approval

Do not:
- switch branches;
- merge, rebase, cherry-pick, or reset;
- commit or push;
- force-push or rewrite history;
- delete branches, tags, files, or directories;
- run destructive cleanup commands;
- overwrite user changes;
- modify Pavan's branch or reference checkout.

Do not assume a clean working tree. Do not drop or rewrite stashes.

## Change discipline

Before editing, list intended files and why each must change. Keep changes limited to the approved Person B scope. Avoid unrelated formatting, dependency, or configuration changes.

After editing, report the exact diff summary and tests. Do not commit or push unless separately authorized.
