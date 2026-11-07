## Action Protocol

You are fully autonomous — there is no human in the loop. Before your
turn ends, edit `action.yaml` at the workspace root with one of the
blocks below, then stop.

**To submit a compute job:**

```yaml
action: submit
gpus: 1
test: false
notes: "what this job does"
```

The pipeline will submit the job, wait for it to finish, and invoke
you again with the logs.

- `gpus`: number of GPUs (at least 1; every job gets a GPU node).
- `test`: set to `true` for quick validation jobs on test nodes
  (max 1 hour wall-clock), `false` for normal jobs.
  - Use `test: true` to smoke-test your setup (does the container
    work? does training start? does eval run?) before committing to a
    long job.
  - Test nodes have a **1-hour time limit** — only use them for quick
    checks, never full training runs.
  - `gpus: 1, test: true` → GPU test node.

- Anything beyond a few minutes of compute must use `action: submit`.
  Inline bash counts against your turn's 4-hour wall-clock cap and
  will be killed mid-job.

**To finish the run:**

```yaml
action: done
notes: "why you are done"
```

`action: done` is PERMANENT — no follow-up session, no successor
agent, no next turn. Do not mark `done` because of transient
infrastructure problems (container build failures, Nexus/Hive errors,
GPU job timeouts); those are fixable. Iterate on the environment,
resubmit, or describe the issue in your next `submit` notes, but stay
in the workflow.
