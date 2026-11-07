"""Background job monitor -- polls Hive, triggers checkpoints, wakes agents.

The :class:`JobMonitor` runs as a daemon thread inside the Nexus daemon
process.  Each tick it polls the Hive dashboard for status updates on
all tracked jobs, writes checkpoint/completion data to agent workspaces,
and invokes per-project wake commands to notify agents of events.

Designed for resilience: errors on individual jobs are logged but never
crash the monitor loop.  On daemon restart, :meth:`JobMonitor.start`
recovers in-flight jobs from disk.
"""

import logging
import subprocess
import threading
import time
from typing import Dict, Optional

from .config import NexusConfig
from .hive_bridge import HiveBridge
from .jobs import NexusJob
from .project import Project

log = logging.getLogger(__name__)


class _TrackedJob:
    """Internal: a job being monitored alongside its parent project."""

    __slots__ = ("job", "project", "preempted_since", "resumed_at")

    def __init__(self, job: NexusJob, project: Project):
        self.job = job
        self.project = project
        self.preempted_since: float = 0.0  # when preempted state first seen
        self.resumed_at: float = 0.0       # when the job was last continued


class JobMonitor:
    """Background thread that monitors active Nexus jobs via Hive polling.

    The monitor maintains an in-memory set of tracked jobs.  Each tick
    (controlled by ``config.monitor.status_poll_interval``):

    1. Polls Hive for each tracked job's latest status.
    2. Detects status transitions (queued -> running -> completed/failed).
    3. Triggers periodic checkpoints (writes logs, wakes agent).
    4. Handles completions (writes results, releases GPUs, records
       GPU-time, wakes agent).
    5. Flags a project ``budget_exhausted`` once it is over its GPU-time
       budget and has no active job left.

    Args:
        bridge: :class:`HiveBridge` instance for Hive HTTP communication.
        config: :class:`NexusConfig` with monitoring intervals.
    """

    def __init__(self, bridge: HiveBridge, config: NexusConfig):
        self.bridge = bridge
        self.config = config
        self._tracked: Dict[str, _TrackedJob] = {}  # run_id -> _TrackedJob
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background monitor thread.

        The thread is a daemon thread -- it will be killed automatically
        when the main process exits.  Safe to call multiple times (noop
        if already running).

        On start, reloads any in-flight jobs from disk so that daemon
        restarts don't leave jobs permanently stuck as "running".
        """
        if self._thread is not None and self._thread.is_alive():
            log.warning("Monitor thread already running")
            return

        # Reload in-flight jobs from disk.  After a daemon restart, jobs
        # recorded as queued, running or awaiting_response would otherwise
        # never be polled again.
        reloaded = 0
        all_jobs = NexusJob.load_all()
        # First pass: re-register active jobs and tally their GPU usage
        # per project for the reconcile step below.
        active_gpus_per_project: dict[str, int] = {}
        for job in all_jobs:
            if job.status in ("queued", "running", "awaiting_response"):
                project = Project.load(job.project_id)
                if project is None:
                    # The job's project was deleted: mark the job failed.
                    job.status = "failed"
                    job.completed_at = time.time()
                    job.save()
                    log.warning("Orphaned job %s (project %s gone), marked failed",
                                job.run_id, job.project_id)
                    continue
                active_gpus_per_project[job.project_id] = (
                    active_gpus_per_project.get(job.project_id, 0)
                    + (job.num_gpus or 0)
                )
                with self._lock:
                    if job.run_id not in self._tracked:
                        self._tracked[job.run_id] = _TrackedJob(job=job, project=project)
                        reloaded += 1
        if reloaded:
            log.info("Reloaded %d in-flight jobs from disk", reloaded)

        # Recompute gpus_in_use from the active jobs, for every project
        # that has a job record.  The counter is stale if the daemon died
        # between acquiring or releasing a slot and saving the matching
        # job record; a stale counter would reject submissions to an idle
        # project.  Recomputing it on every start is idempotent.
        seen_projects: set[str] = set()
        for job in all_jobs:
            if job.project_id in seen_projects:
                continue
            seen_projects.add(job.project_id)
            project = Project.load(job.project_id)
            if project is None:
                continue
            expected = active_gpus_per_project.get(job.project_id, 0)
            if project.gpus_in_use != expected:
                log.warning(
                    "Reconciling project %s gpus_in_use: %d -> %d (drift recovered from active jobs)",
                    job.project_id, project.gpus_in_use, expected,
                )
                project.gpus_in_use = expected
                project.save()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="nexus-job-monitor")
        self._thread.start()
        log.info("Job monitor started (poll interval=%ds)",
                 self.config.monitor.status_poll_interval)

    def stop(self):
        """Signal the monitor to stop and wait for thread exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                log.warning("Monitor thread did not exit within timeout")
            self._thread = None
        log.info("Job monitor stopped")

    @property
    def is_running(self) -> bool:
        """Whether the monitor thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Job tracking
    # ------------------------------------------------------------------

    def add_job(self, job: NexusJob, project: Project):
        """Register a newly submitted job for monitoring.

        Args:
            job: The :class:`NexusJob` that was just submitted.
            project: The :class:`Project` the job belongs to.
        """
        with self._lock:
            self._tracked[job.run_id] = _TrackedJob(job=job, project=project)
        log.info("Tracking job %s (project=%s, hive_id=%s)",
                 job.run_id, project.project_id, job.hive_job_id)

    def continue_job(self, run_id: str):
        """Agent says continue after checkpoint.  Resume monitoring.

        Transitions the job from ``awaiting_response`` back to
        ``running`` so the next tick will resume Hive polling.

        Args:
            run_id: The Nexus run ID.
        """
        with self._lock:
            tracked = self._tracked.get(run_id)
            if tracked and tracked.job.status == "awaiting_response":
                tracked.job.status = "running"
                tracked.resumed_at = time.time()
                tracked.job.save()
                log.info("Job %s: agent continued after checkpoint", run_id)
            elif tracked:
                log.debug("Job %s: continue called but status is %s",
                          run_id, tracked.job.status)
            else:
                log.warning("Job %s: continue called but not tracked", run_id)

    def cancel_job(self, run_id: str):
        """Cancel a job.  Tell Hive, update status, release GPUs.

        This is safe to call from any thread.  If the job is not tracked
        (already completed or never registered), this is a noop.  The job
        is removed from tracking *first* so the monitor thread won't race
        with the cancel.

        Args:
            run_id: The Nexus run ID.
        """
        with self._lock:
            tracked = self._tracked.pop(run_id, None)
            if not tracked:
                log.debug("Job %s: cancel called but not tracked", run_id)
                return

        # From here, the monitor thread can no longer see this job.
        job = tracked.job
        project = tracked.project

        # Already terminal: "completed" or "failed" means _handle_completion
        # saw the job end and released its GPUs; "cancelled" means an
        # earlier cancel_job call did.  Releasing again would double-count.
        if job.status in ("completed", "failed", "cancelled"):
            log.debug("Job %s: cancel called but already %s", run_id, job.status)
            return

        # Cancel on Hive.  If that fails the job is still marked cancelled
        # here so that it does not hold quota, although it may keep
        # running on Hive.
        if job.hive_job_id:
            if not self.bridge.cancel(job.hive_job_id):
                log.warning("Hive cancel failed for %s (hive_job=%s); "
                            "marking it cancelled in Nexus anyway",
                            run_id, job.hive_job_id)

        job.status = "cancelled"
        job.completed_at = time.time()

        # Release GPUs before persisting the terminal status, for the same
        # reason as in _handle_completion: the wait endpoint returns as
        # soon as it sees the terminal status on disk, and a submission
        # that follows must already find the slot free.
        project.release_gpus(job.num_gpus)

        job.save()

        # Fetch final logs from Hive for the workspace.  A failed poll
        # carries no logs: keep the files of the last checkpoint then,
        # rather than emptying them.
        if job.hive_job_id:
            try:
                result = self.bridge.poll(job.hive_job_id)
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                if project.workspace_dir and result.get("status") != "unknown":
                    job.write_logs(project.workspace_dir, stdout, stderr)
            except Exception as exc:
                log.debug("Could not fetch final logs for cancelled job %s: %s",
                          run_id, exc)
        log.info("Job %s cancelled, released %d GPU(s) for project %s",
                 run_id, job.num_gpus, project.project_id)

    def tracked_jobs(self) -> Dict[str, _TrackedJob]:
        """Return a snapshot of all currently tracked jobs.

        Returns:
            A dict mapping run_id to ``_TrackedJob``.  This is a shallow
            copy -- callers should not mutate the returned objects.
        """
        with self._lock:
            return dict(self._tracked)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self):
        """Main monitor loop.  Runs until ``_stop_event`` is set."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("Monitor tick error")
            # Wait for the poll interval (or until stop is signalled)
            self._stop_event.wait(
                timeout=self.config.monitor.status_poll_interval,
            )

    def _tick(self):
        """Single monitor iteration: poll all tracked jobs."""
        with self._lock:
            snapshot = list(self._tracked.values())

        for tracked in snapshot:
            try:
                self._check_job(tracked)
            except Exception:
                log.exception("Error checking job %s", tracked.job.run_id)

    # ------------------------------------------------------------------
    # Per-job checking
    # ------------------------------------------------------------------

    def _check_job(self, tracked: _TrackedJob):
        """Check a single job's status and act accordingly.

        Handles three states:
        - **awaiting_response**: check for checkpoint timeout
          (auto-continue).
        - **queued/running**: poll Hive, detect transitions, trigger
          checkpoints or handle completion.
        """
        job = tracked.job
        project = tracked.project

        # Terminal states should have been removed already.
        if job.status in ("completed", "failed", "cancelled"):
            return

        # While awaiting agent response, don't poll Hive -- just check
        # for checkpoint timeout (auto-continue).
        if job.status == "awaiting_response":
            self._check_checkpoint_timeout(tracked)
            return

        # Not yet submitted to Hive (shouldn't happen, but be safe).
        if not job.hive_job_id:
            return

        # Poll Hive for latest status.
        result = self.bridge.poll(job.hive_job_id)
        hive_status = result.get("status", "unknown")

        if hive_status == "unknown":
            return  # Transient error -- try again next tick.

        # Detect queued -> running transition.
        if job.status == "queued" and hive_status in ("running",):
            job.status = "running"
            job.started_at = time.time()
            job.save()
            if project.workspace_dir:
                job.write_meta(project.workspace_dir)
            log.info("Job %s is now running on Hive", job.run_id)

        # Detect completion (completed, failed, timeout, cancelled).
        # "preempted" is transient: Hive re-queues the job by itself.  It
        # is treated as terminal only after more than 10 minutes in that
        # state.
        if hive_status != "preempted":
            tracked.preempted_since = 0.0  # reset when status clears
        if hive_status == "preempted":
            if not tracked.preempted_since:
                tracked.preempted_since = time.time()
            preempt_age = time.time() - tracked.preempted_since
            if preempt_age > 600:
                log.warning("Job %s stuck in preempted for %.0fs, treating as failed",
                            job.run_id, preempt_age)
                self._handle_completion(tracked, result)
            return
        if hive_status in ("completed", "failed", "timeout", "cancelled"):
            self._handle_completion(tracked, result)
            return

        # While running, check if it's time for a checkpoint.
        if job.status == "running":
            self._maybe_checkpoint(tracked, result)

    def _check_checkpoint_timeout(self, tracked: _TrackedJob):
        """Auto-continue a job if the agent hasn't responded in time.

        Uses the per-project checkpoint_timeout override if set,
        otherwise falls back to the global config value.
        """
        job = tracked.job
        project = tracked.project

        checkpoint_timeout = (
            project.checkpoint_timeout
            or self.config.monitor.checkpoint_timeout
        )

        if (job.last_checkpoint_at
                and time.time() - job.last_checkpoint_at > checkpoint_timeout):
            log.info("Job %s: checkpoint timeout (%ds), auto-continuing",
                     job.run_id, checkpoint_timeout)
            job.status = "running"
            tracked.resumed_at = time.time()
            job.save()

    def _maybe_checkpoint(self, tracked: _TrackedJob, hive_result: dict):
        """Trigger a checkpoint if enough time has elapsed since the last one.

        Uses the per-project checkpoint_interval override if set,
        otherwise falls back to the global config value.
        """
        job = tracked.job
        project = tracked.project

        checkpoint_interval = (
            project.checkpoint_interval
            or self.config.monitor.checkpoint_interval
        )

        # The interval counts from the last checkpoint or, if the job was
        # continued since, from that moment. Otherwise a job that was
        # continued after more than one interval (the automatic continue
        # always is) would be checkpointed again on the next tick.
        last_cp = max(job.last_checkpoint_at or 0.0, tracked.resumed_at) \
            or job.started_at or job.submitted_at
        if time.time() - last_cp >= checkpoint_interval:
            self._handle_checkpoint(tracked, hive_result)

    # ------------------------------------------------------------------
    # Completion handling
    # ------------------------------------------------------------------

    def _handle_completion(self, tracked: _TrackedJob, hive_result: dict):
        """Job completed/failed on Hive.  Write results, release GPUs, wake agent.

        Determines final status from the Hive result, writes all output
        files to the workspace, releases GPUs back to the project quota,
        wakes the agent, and removes the job from tracking.

        Safe against races: if cancel_job already popped this job from
        tracking, the pop here is a harmless noop.
        """
        job = tracked.job
        project = tracked.project

        # Guard: cancel_job may have already handled this job.
        if job.status in ("cancelled",):
            with self._lock:
                self._tracked.pop(job.run_id, None)
            return

        stdout = hive_result.get("stdout", "")
        stderr = hive_result.get("stderr", "")
        hive_status = hive_result.get("status", "failed")

        returncode = hive_result.get("returncode")
        if returncode is None:
            if hive_status == "completed":
                # Hive reports completion without a return code: assume success.
                log.warning("Job %s: Hive status is 'completed' but no returncode; assuming 0",
                            job.run_id)
                returncode = 0
            else:
                returncode = -1

        # Record the Hive resource the job actually ran on.
        actual = hive_result.get("resource_name")
        if actual:
            job.actual_resource = actual

        # Backfill started_at if the job completed before we saw it running.
        # Use Hive's started_at if available, otherwise use submitted_at.
        if job.started_at is None:
            job.started_at = (
                hive_result.get("started_at")
                or job.submitted_at
            )

        # Determine final Nexus status.
        if hive_status == "completed" and returncode == 0:
            job.status = "completed"
        else:
            job.status = "failed"

        job.completed_at = time.time()
        job.returncode = returncode

        # Release GPUs before persisting the terminal status.  A client
        # blocked on the wait endpoint returns as soon as it sees the
        # terminal status on disk and may submit its next job at once.
        # If the slot were released only after the workspace writes below
        # (which can be slow on a network filesystem), that submission
        # could be rejected for a slot that is in fact free.
        # Order: release, save, write.
        project.release_gpus(job.num_gpus)

        job.save()

        # Write all output to the agent workspace.
        if project.workspace_dir:
            try:
                job.write_logs(project.workspace_dir, stdout, stderr)
                job.write_result(project.workspace_dir, stdout, stderr, returncode)
                job.write_meta(project.workspace_dir)
            except OSError as exc:
                log.warning("Could not write workspace files for %s: %s",
                            job.run_id, exc)

        # Record GPU-time usage for soft budget tracking.
        if job.started_at is not None:
            elapsed = (job.completed_at or time.time()) - job.started_at
            project = Project.load(project.project_id) or project
            project.record_gpu_usage(job.num_gpus, max(0, elapsed))

        # Wake agent with completion event.
        event = "completed" if job.status == "completed" else "failed"
        self._wake_agent(project, job, event)

        log.info("Job %s %s (rc=%s), released %d GPU(s) for project %s",
                 job.run_id, job.status, returncode, job.num_gpus,
                 project.project_id)

        # Remove from tracking (may already be gone if cancel_job ran).
        with self._lock:
            self._tracked.pop(job.run_id, None)

        # Check soft GPU budget after removing this job from tracking.
        self._check_soft_budget(project)

    # ------------------------------------------------------------------
    # Checkpoint handling
    # ------------------------------------------------------------------

    def _handle_checkpoint(self, tracked: _TrackedJob, hive_result: dict):
        """Periodic checkpoint: write logs, wake agent, set awaiting_response.

        Writes a numbered checkpoint file and the latest logs to the
        workspace, then transitions the job to ``awaiting_response`` and
        wakes the agent.  The agent can respond with ``continue`` or
        ``cancel``; if it doesn't respond within the checkpoint timeout,
        the monitor auto-continues.
        """
        job = tracked.job
        project = tracked.project

        stdout = hive_result.get("stdout", "")
        stderr = hive_result.get("stderr", "")

        # Guard: cancel_job may have already handled this job.
        if job.status in ("cancelled", "completed", "failed"):
            return

        # Write checkpoint data to workspace.
        if project.workspace_dir:
            try:
                job.write_checkpoint(project.workspace_dir, stdout, stderr)
                job.write_logs(project.workspace_dir, stdout, stderr)
            except OSError as exc:
                log.warning("Could not write checkpoint for %s: %s",
                            job.run_id, exc)
        job.last_checkpoint_at = time.time()
        job.status = "awaiting_response"
        job.save()

        # Wake agent.
        self._wake_agent(project, job, "checkpoint")

        log.info("Job %s: checkpoint #%d, awaiting agent response",
                 job.run_id, job.checkpoint_count)

    # ------------------------------------------------------------------
    # Agent wake mechanism
    # ------------------------------------------------------------------

    def _wake_agent(self, project: Project, job: NexusJob, event: str):
        """Execute the project's wake_command to notify the agent.

        Template variables in the wake command are expanded:
        ``{project_id}``, ``{run_id}``, ``{event}``, ``{workspace}``,
        ``{jobs_dir}``.

        If the wake command fails or times out, this logs a warning but
        does not raise -- the agent may still pick up the event via
        polling or on its next ``nx wait``.

        Args:
            project: The project whose agent should be woken.
            job: The job that triggered the wake event.
            event: One of ``checkpoint``, ``completed``, ``failed``,
                ``budget_exhausted``.
        """
        if not project.wake_command:
            return

        jobs_dir = f"{project.workspace_dir}/jobs/{job.run_id}"
        try:
            cmd = project.wake_command.format(
                project_id=project.project_id,
                run_id=job.run_id,
                event=event,
                workspace=project.workspace_dir,
                jobs_dir=jobs_dir,
            )
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            # A malformed template must not escape: _handle_completion would
            # then never stop tracking the job.
            log.warning("Wake command template error for project %s: %r",
                        project.project_id, exc)
            return

        try:
            subprocess.run(
                cmd,
                shell=True,
                timeout=30,
                capture_output=True,
            )
            log.debug("Wake command executed for %s/%s event=%s",
                      project.project_id, job.run_id, event)
        except subprocess.TimeoutExpired:
            log.warning("Wake command timed out for %s/%s event=%s",
                        project.project_id, job.run_id, event)
        except Exception as exc:
            log.warning("Wake command failed for %s/%s event=%s: %s",
                        project.project_id, job.run_id, event, exc)

    # ------------------------------------------------------------------
    # Soft GPU budget
    # ------------------------------------------------------------------

    def _check_soft_budget(self, project: Project):
        """Check if a project has exceeded its soft GPU budget.

        Called after a job completes.  If the budget is exceeded and no
        other jobs for this project are still running, sets
        ``budget_exhausted = True`` and wakes the agent.
        """
        project = Project.load(project.project_id) or project
        if not project.is_over_soft_budget():
            return

        # Check if any jobs for this project are still running.
        with self._lock:
            still_running = any(
                t.project.project_id == project.project_id
                for t in self._tracked.values()
            )

        if still_running:
            log.info("Project %s over soft GPU budget but jobs still running",
                     project.project_id)
            return

        project.budget_exhausted = True
        project.save()
        log.info("Project %s: soft GPU budget exhausted (%.1f / %.1f gpu-hours)",
                 project.project_id,
                 project.gpu_seconds_used / 3600,
                 project.soft_gpu_budget_hours)

        # Wake agent with budget_exhausted event using a synthetic job.
        dummy_job = NexusJob(project_id=project.project_id, run_id="budget")
        self._wake_agent(project, dummy_job, "budget_exhausted")
