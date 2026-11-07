import json
import threading
import time
from collections import deque
from pathlib import Path

from swarm.scheduler import (
    PaperJob,
    Scheduler,
    _base_sif_cache_filename,
    _can_rewrite_def_to_localimage,
    _canonical_docker_ref,
    _classify_env_failure_text,
    _extract_container_source_uri,
    _is_model_limit_error_text,
    _iteration_completion_topics,
    _parse_docker_image_ref,
    _rewrite_def_to_localimage,
)


def test_classify_env_failure_name_unknown():
    failure_class, signature, hint = _classify_env_failure_text(
        "FATAL: NAME_UNKNOWN: repository with name 'cuda' does not exist in the registry",
        "build",
    )
    assert failure_class == "invalid_spec"
    assert signature == "container image not found"
    assert "base image reference" in hint


def test_classify_env_failure_registry_parse():
    failure_class, signature, hint = _classify_env_failure_text(
        "conveyor failed to get: invalid character 's' after top-level value",
        "build",
    )
    assert failure_class == "transient_external"
    assert signature == "registry response malformed"
    assert "registry or network" in hint


def test_classify_env_failure_disk_quota():
    failure_class, signature, hint = _classify_env_failure_text(
        "[Errno 122] Disk quota exceeded",
        "setup",
    )
    assert failure_class == "system_capacity"
    assert signature == "disk quota exceeded"


def test_is_model_limit_error_text_matches():
    assert _is_model_limit_error_text(
        'rc=1: API Error: 429 {"error":{"code":"1310",'
        '"message":"Weekly/Monthly Limit Exhausted."}}'
    )
    assert _is_model_limit_error_text("usage limit reached for this key")
    assert not _is_model_limit_error_text("rate limit exceeded, retrying")
    assert not _is_model_limit_error_text("worker exited with error status")


def test_extract_container_source_uri_for_docker(tmp_path: Path):
    container_def = tmp_path / "container.def"
    container_def.write_text(
        "Bootstrap: docker\n"
        "From: public.ecr.aws/nvidia/cuda:12.1.0-runtime-ubuntu22.04\n"
    )
    assert (
        _extract_container_source_uri(container_def)
        == "docker://public.ecr.aws/nvidia/cuda:12.1.0-runtime-ubuntu22.04"
    )


def test_extract_container_source_uri_skips_unsupported_bootstrap(tmp_path: Path):
    container_def = tmp_path / "container.def"
    container_def.write_text(
        "Bootstrap: localimage\n"
        "From: /tmp/base.sif\n"
    )
    assert _extract_container_source_uri(container_def) is None


def test_extract_container_source_uri_honors_registry_namespace(tmp_path: Path):
    # A bare ``From`` combined with Registry/Namespace headers must
    # produce the canonical fully-qualified docker URI so prevalidation
    # inspects the actual image, not ``docker://image:v1`` against
    # Docker Hub.
    container_def = tmp_path / "container.def"
    container_def.write_text(
        "Bootstrap: docker\n"
        "Registry: ghcr.io\n"
        "Namespace: owner\n"
        "From: image:v1\n"
    )
    assert (
        _extract_container_source_uri(container_def)
        == "docker://ghcr.io/owner/image:v1"
    )


def test_extract_container_source_uri_skips_ambiguous_registry_merge(tmp_path: Path):
    # When the merge is ambiguous (slashed From + Registry header), we
    # would rather skip prevalidation than validate the wrong image.
    container_def = tmp_path / "container.def"
    container_def.write_text(
        "Bootstrap: docker\n"
        "Registry: ghcr.io\n"
        "From: nvidia/cuda:12.1\n"
    )
    assert _extract_container_source_uri(container_def) is None


def test_extract_container_source_uri_skips_multi_stage_def(tmp_path: Path):
    # A multi-stage def has multiple Bootstrap headers. A Registry or
    # Namespace from a later stage must NEVER be merged into the first
    # stage's From, so the safe behavior is to skip prevalidation.
    container_def = tmp_path / "container.def"
    container_def.write_text(
        "Bootstrap: docker\n"
        "From: python:3.11\n"
        "Stage: build\n"
        "\n"
        "Bootstrap: docker\n"
        "Registry: ghcr.io\n"
        "From: alpine:3.19\n"
        "Stage: final\n"
    )
    assert _extract_container_source_uri(container_def) is None


# ── Base SIF cache helpers ────────────────────────────────────────────


def test_parse_docker_image_ref_simple():
    assert _parse_docker_image_ref("python:3.11-slim") == {
        "domain": "docker.io",
        "name": "library/python",
        "tag": "3.11-slim",
        "digest": None,
    }


def test_parse_docker_image_ref_namespaced():
    assert _parse_docker_image_ref("nvidia/cuda:12.1.0-devel") == {
        "domain": "docker.io",
        "name": "nvidia/cuda",
        "tag": "12.1.0-devel",
        "digest": None,
    }


def test_parse_docker_image_ref_explicit_registry():
    assert _parse_docker_image_ref("public.ecr.aws/nvidia/cuda:12.1.0-runtime-ubuntu22.04") == {
        "domain": "public.ecr.aws",
        "name": "nvidia/cuda",
        "tag": "12.1.0-runtime-ubuntu22.04",
        "digest": None,
    }


def test_parse_docker_image_ref_localhost_with_port():
    assert _parse_docker_image_ref("localhost:5000/myimage:tag") == {
        "domain": "localhost:5000",
        "name": "myimage",
        "tag": "tag",
        "digest": None,
    }


def test_parse_docker_image_ref_digest_pin():
    assert _parse_docker_image_ref("python:3.11@sha256:abc123") == {
        "domain": "docker.io",
        "name": "library/python",
        "tag": "3.11",
        "digest": "sha256:abc123",
    }


def test_parse_docker_image_ref_no_tag():
    parsed = _parse_docker_image_ref("ubuntu")
    assert parsed == {
        "domain": "docker.io",
        "name": "library/ubuntu",
        "tag": None,
        "digest": None,
    }


def test_parse_docker_image_ref_rejects_invalid():
    assert _parse_docker_image_ref("") is None
    assert _parse_docker_image_ref("   ") is None
    assert _parse_docker_image_ref("bad ref with spaces") is None
    assert _parse_docker_image_ref("python@") is None


def test_canonical_docker_ref_default_tag_for_bare_name():
    assert _canonical_docker_ref("python", None, None) == "docker.io/library/python:latest"


def test_canonical_docker_ref_preserves_explicit_tag():
    assert _canonical_docker_ref("python:3.11-slim", None, None) == "docker.io/library/python:3.11-slim"


def test_canonical_docker_ref_explicit_registry_in_from():
    assert (
        _canonical_docker_ref("ghcr.io/owner/repo:v1", None, None)
        == "ghcr.io/owner/repo:v1"
    )


def test_canonical_docker_ref_registry_header_overrides_default():
    assert (
        _canonical_docker_ref("python:3.11", "ghcr.io", "myorg")
        == "ghcr.io/myorg/python:3.11"
    )


def test_canonical_docker_ref_strips_http_scheme_from_registry():
    assert (
        _canonical_docker_ref("python", "http://my.reg.com:5000", "ns")
        == "my.reg.com:5000/ns/python:latest"
    )


def test_canonical_docker_ref_ambiguous_when_from_has_slash_and_registry_set():
    # Setting both Registry and a slashed From is ambiguous — fall through.
    assert _canonical_docker_ref("nvidia/cuda:12.1", "ghcr.io", None) is None


def test_canonical_docker_ref_ambiguous_when_from_has_port_qualified_registry():
    # ``localhost:5000/myimage:tag`` carries its own domain — combining it
    # with a header Registry would silently rewrite to a different image,
    # so we must refuse the merge and fall through.
    assert _canonical_docker_ref("localhost:5000/myimage:tag", "other.reg", None) is None
    assert _canonical_docker_ref("my.reg.com:5000/foo:1", None, "ns") is None


def test_canonical_docker_ref_ambiguous_when_from_has_explicit_namespace():
    # ``myorg/foo`` resolves on docker.io but uses an explicit namespace,
    # so honoring a header Namespace on top of it is ambiguous.
    assert _canonical_docker_ref("myorg/foo:1", None, "otherns") is None


def test_canonical_docker_ref_ambiguous_when_from_uses_explicit_docker_io():
    # ``docker.io/python`` and ``library/python`` are explicit-domain /
    # explicit-namespace forms of the same image as plain ``python``, but
    # combining them with a header Registry/Namespace must still be
    # treated as ambiguous (the parser normalizes all three to the same
    # shape, so the ambiguity check must run on the raw From string).
    assert _canonical_docker_ref("docker.io/python", "ghcr.io", None) is None
    assert _canonical_docker_ref("docker.io/python:3.11", None, "myorg") is None
    assert _canonical_docker_ref("library/python", "ghcr.io", None) is None
    assert _canonical_docker_ref("library/python:3.11", None, "myorg") is None


def test_canonical_docker_ref_digest_pin_omits_tag():
    assert (
        _canonical_docker_ref("python:3.11@sha256:abc", None, None)
        == "docker.io/library/python@sha256:abc"
    )


def test_base_sif_cache_filename_is_unique_per_canonical():
    fn1 = _base_sif_cache_filename("docker.io/library/python:3.11")
    fn2 = _base_sif_cache_filename("docker.io/library/python:3.12")
    fn3 = _base_sif_cache_filename("docker.io/library/python:3.11")
    assert fn1 != fn2
    assert fn1 == fn3
    assert fn1.endswith(".sif")
    assert "/" not in fn1
    assert ":" not in fn1


def test_can_rewrite_simple_docker_def():
    safe, canonical = _can_rewrite_def_to_localimage(
        "Bootstrap: docker\n"
        "From: python:3.11-slim\n"
        "\n"
        "%post\n"
        "    echo hello\n"
    )
    assert safe is True
    assert canonical == "docker.io/library/python:3.11-slim"


def test_can_rewrite_skips_multi_stage():
    safe, canonical = _can_rewrite_def_to_localimage(
        "Bootstrap: docker\n"
        "From: golang:1.20\n"
        "Stage: build\n"
        "\n"
        "Bootstrap: docker\n"
        "From: alpine:latest\n"
    )
    assert safe is False
    assert canonical is None


def test_can_rewrite_skips_localimage_bootstrap():
    safe, _ = _can_rewrite_def_to_localimage(
        "Bootstrap: localimage\n"
        "From: /tmp/base.sif\n"
    )
    assert safe is False


def test_can_rewrite_handles_registry_namespace_for_bare_image():
    safe, canonical = _can_rewrite_def_to_localimage(
        "Bootstrap: docker\n"
        "Registry: ghcr.io\n"
        "Namespace: owner\n"
        "From: image:v1\n"
    )
    assert safe is True
    assert canonical == "ghcr.io/owner/image:v1"


def test_can_rewrite_skips_unknown_header_keys():
    # ``Fingerprint:`` is a real Apptainer header we don't understand —
    # safer to skip than guess.
    safe, _ = _can_rewrite_def_to_localimage(
        "Bootstrap: docker\n"
        "From: python:3.11\n"
        "Fingerprint: AAAA\n"
    )
    assert safe is False


def test_can_rewrite_handles_inline_comments():
    safe, canonical = _can_rewrite_def_to_localimage(
        "Bootstrap: docker\n"
        "From: python:3.11  # base image\n"
    )
    assert safe is True
    assert canonical == "docker.io/library/python:3.11"


def test_rewrite_def_replaces_bootstrap_and_from(tmp_path: Path):
    src = (
        "# Pytorch base\n"
        "Bootstrap: docker\n"
        "From: pytorch/pytorch:2.4.0-cuda12.1\n"
        "\n"
        "%files\n"
        "    setup.sh /setup.sh\n"
        "\n"
        "%post\n"
        "    apt-get update\n"
    )
    sif = tmp_path / "pytorch_2.4.0.sif"
    out = _rewrite_def_to_localimage(src, sif)
    assert out is not None
    assert "Bootstrap: localimage" in out
    assert f"From: {sif}" in out
    # Body sections preserved verbatim.
    assert "%files" in out
    assert "setup.sh /setup.sh" in out
    assert "%post" in out
    assert "apt-get update" in out
    # Original docker reference is no longer the active From.
    assert "From: pytorch/pytorch:2.4.0-cuda12.1" not in out
    # Leading documentation comment is preserved.
    assert out.lstrip().startswith("# Pytorch base")


def test_rewrite_def_returns_none_when_unsafe(tmp_path: Path):
    sif = tmp_path / "x.sif"
    assert _rewrite_def_to_localimage(
        "Bootstrap: localimage\nFrom: /tmp/base.sif\n", sif,
    ) is None


def test_iteration_completion_topics_for_intermediate_done_iteration():
    assert _iteration_completion_topics(
        3,
        status="done",
        has_next_iteration=True,
    ) == ["iteration-3-complete"]


def test_iteration_completion_topics_for_final_done_iteration():
    assert _iteration_completion_topics(
        4,
        status="done",
        has_next_iteration=False,
    ) == ["iteration-4-complete", "done"]


def test_iteration_completion_topics_skip_non_done_iterations():
    assert _iteration_completion_topics(
        2,
        status="error",
        has_next_iteration=False,
    ) == []


def test_recover_all_papers_recovers_everything_then_dispatches_once(monkeypatch, tmp_path: Path):
    papers_dir = tmp_path / "papers"
    (papers_dir / "a1").mkdir(parents=True)
    (papers_dir / "a2").mkdir(parents=True)
    (papers_dir / "skip").mkdir(parents=True)
    (papers_dir / "a1" / "meta.json").write_text(json.dumps({
        "arxiv_id": "a1",
        "error": "daemon restarted — use 'swarm continue-all' to resume",
        "recovery_action": "continue",
    }))
    (papers_dir / "a2" / "meta.json").write_text(json.dumps({
        "arxiv_id": "a2",
        "error": "daemon restarted — use 'swarm continue-all' to resume",
        "recovery_action": "submit",
    }))
    (papers_dir / "skip" / "meta.json").write_text(json.dumps({
        "arxiv_id": "skip",
        "error": "some other error",
    }))

    monkeypatch.setattr("swarm.scheduler.PAPERS_DIR", papers_dir)

    scheduler = Scheduler.__new__(Scheduler)
    calls: list[tuple[str, bool, bool]] = []
    events: list[str] = []
    scheduler._lock = threading.Lock()
    scheduler.active = {}
    scheduler.queue = []

    def fake_recover_paper(arxiv_id, profile=None, no_workflow=False, fresh=False,
                           dispatch=True, skip_github=False):
        calls.append((arxiv_id, dispatch, skip_github))
        events.append(f"recover:{arxiv_id}")
        scheduler.queue.append(arxiv_id)
        return PaperJob(arxiv_id=arxiv_id, paper=None, workspace_path="", gpu_budget=1.0, status="queued")

    scheduler.recover_paper = fake_recover_paper
    def fake_dispatch_next():
        if scheduler.queue:
            arxiv_id = scheduler.queue.pop(0)
            events.append(f"dispatch:{arxiv_id}")
            scheduler.active[str(len(scheduler.active) + 1)] = object()
            return True
        return False
    scheduler._dispatch_next = fake_dispatch_next

    result = Scheduler.recover_all_papers(scheduler, profile="p")

    assert result["total"] == 2
    assert result["queued"] == 2
    assert result["failed"] == 0
    # Each paper is re-queued without dispatching and without GitHub calls
    # (both are deferred to the worker thread) ...
    assert calls == [("a1", False, True), ("a2", False, True)]
    # ... and a single dispatch round runs once everything is queued. The
    # periodic dispatch tick admits the rest.
    assert events == [
        "recover:a1",
        "recover:a2",
        "dispatch:a1",
    ]
    assert [r["recovery_action"] for r in result["results"]] == ["continue", "submit"]


def test_recover_all_papers_reports_failures(monkeypatch, tmp_path: Path):
    papers_dir = tmp_path / "papers"
    (papers_dir / "a1").mkdir(parents=True)
    (papers_dir / "a1" / "meta.json").write_text(json.dumps({
        "arxiv_id": "a1",
        "error": "daemon restarted — use 'swarm continue-all' to resume",
        "recovery_action": "continue",
    }))
    monkeypatch.setattr("swarm.scheduler.PAPERS_DIR", papers_dir)

    scheduler = Scheduler.__new__(Scheduler)

    def failing_recover_paper(arxiv_id, **kwargs):
        raise FileNotFoundError(f"No workspace found for {arxiv_id}")

    scheduler.recover_paper = failing_recover_paper
    scheduler._dispatch_next = lambda: False

    result = Scheduler.recover_all_papers(scheduler)

    assert result["total"] == 1
    assert result["queued"] == 0
    assert result["failed"] == 1
    assert result["results"][0]["status"] == "failed"
    assert "No workspace found" in result["results"][0]["error"]


class _DummyConfig:
    def split_runtime_caps_enabled(self): return True
    def effective_max_working_workers(self): return 2
    def effective_max_working_return_overflow(self): return 1
    def effective_max_admitted_workers(self): return 1
    def effective_max_building_workers(self): return 1
    def effective_max_installing_workers(self): return 1
    def effective_max_runtime_workers(self): return 3


class _LegacyConfig:
    def split_runtime_caps_enabled(self): return False
    def effective_max_working_workers(self): return 2
    def effective_max_working_return_overflow(self): return 0
    def effective_max_admitted_workers(self): return 3
    def effective_max_building_workers(self): return 1
    def effective_max_installing_workers(self): return 1
    def effective_max_runtime_workers(self): return 3


def _make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._lock = threading.Lock()
    scheduler._capacity_cv = threading.Condition(scheduler._lock)
    scheduler._capacity_state = {}
    scheduler._cap_overrides = {}
    scheduler._working_priority_waiters = deque()
    scheduler._working_fresh_waiters = deque()
    scheduler._working_waiter_kind = {}
    scheduler._shutdown = False
    scheduler.active = {}
    scheduler.queue = deque()
    scheduler.completed = {}
    scheduler._max_total_workers = lambda: 10
    scheduler._save_paper_meta = lambda job: None
    scheduler._launch_calls = 0
    scheduler._launch_dispatched_job = lambda job: setattr(scheduler, "_launch_calls", scheduler._launch_calls + 1)
    scheduler.config = _DummyConfig()
    return scheduler


def test_set_worker_status_releases_current_slot_while_waiting_for_next():
    scheduler = _make_scheduler()
    job = PaperJob(arxiv_id="p1", paper=None, workspace_path="", gpu_budget=1.0)
    job.worker_status = "working"
    job.worker_status_since = 1.0
    scheduler.active = {"p1": job}
    scheduler._capacity_state = {"p1": "working", "b1": "building"}

    t = threading.Thread(target=Scheduler._set_worker_status, args=(scheduler, job, "building"))
    t.start()
    time.sleep(0.1)

    assert job.worker_status == "working"
    assert job.blocked_on == "building"
    assert scheduler._capacity_state.get("p1") is None

    with scheduler._capacity_cv:
        scheduler._capacity_state.pop("b1", None)
        scheduler._capacity_cv.notify_all()
    t.join(timeout=2)

    assert not t.is_alive()
    assert job.worker_status == "building"
    assert job.blocked_on is None
    assert scheduler._capacity_state["p1"] == "building"


def test_return_to_working_has_priority_over_fresh_idle():
    scheduler = _make_scheduler()
    returning = PaperJob(arxiv_id="p1", paper=None, workspace_path="", gpu_budget=1.0)
    fresh = PaperJob(arxiv_id="p2", paper=None, workspace_path="", gpu_budget=1.0)
    returning.worker_status = "waiting"
    returning.worker_status_since = 1.0
    fresh.worker_status = "idle"
    fresh.worker_status_since = 1.0
    scheduler.active = {"p1": returning, "p2": fresh}
    scheduler._capacity_state = {
        "p1": "waiting",
        "p2": "admitted",
        "w1": "working",
        "w2": "working",
        "w3": "working",
    }

    fresh_thread = threading.Thread(target=Scheduler._set_worker_status, args=(scheduler, fresh, "working"))
    return_thread = threading.Thread(target=Scheduler._set_worker_status, args=(scheduler, returning, "working"))
    fresh_thread.start()
    time.sleep(0.05)
    return_thread.start()
    time.sleep(0.1)

    assert fresh.blocked_on == "working"
    assert returning.blocked_on == "working"

    with scheduler._capacity_cv:
        scheduler._capacity_state.pop("w3", None)
        scheduler._capacity_cv.notify_all()

    return_thread.join(timeout=2)
    assert not return_thread.is_alive()
    assert returning.worker_status == "working"
    assert returning.blocked_on is None
    assert scheduler._capacity_state["p1"] == "working"

    assert fresh_thread.is_alive()
    assert fresh.worker_status == "idle"
    assert fresh.blocked_on == "working"

    with scheduler._capacity_cv:
        scheduler._capacity_state.pop("w1", None)
        scheduler._capacity_state.pop("w2", None)
        scheduler._capacity_cv.notify_all()

    fresh_thread.join(timeout=2)
    assert not fresh_thread.is_alive()
    assert fresh.worker_status == "working"
    assert fresh.blocked_on is None


def test_priority_returners_can_use_working_overflow():
    scheduler = _make_scheduler()
    returning = PaperJob(arxiv_id="p1", paper=None, workspace_path="", gpu_budget=1.0)
    fresh = PaperJob(arxiv_id="p2", paper=None, workspace_path="", gpu_budget=1.0)
    returning.worker_status = "waiting"
    fresh.worker_status = "idle"
    scheduler._capacity_state = {"w1": "working", "w2": "working"}
    scheduler._working_priority_waiters = deque(["p1"])
    scheduler._working_waiter_kind = {"p1": "priority"}

    assert Scheduler._status_can_enter_locked(scheduler, returning, "working", old_status="waiting") is True
    assert Scheduler._status_can_enter_locked(scheduler, fresh, "working", old_status="idle") is False


def test_set_worker_status_backfills_queue_when_admitted_slot_frees():
    scheduler = _make_scheduler()
    job = PaperJob(arxiv_id="p1", paper=None, workspace_path="", gpu_budget=1.0)
    queued = PaperJob(arxiv_id="p2", paper=None, workspace_path="", gpu_budget=1.0)
    scheduler.queue = deque([queued])
    scheduler.active = {"p1": job}
    scheduler._capacity_state = {"p1": "admitted"}
    job.worker_status = "idle"
    job.worker_status_since = 1.0

    Scheduler._set_worker_status(scheduler, job, "working")

    assert scheduler._launch_calls == 1
    assert queued.status == "active"
    assert scheduler._capacity_state["p2"] == "admitted"


def test_admitted_holds_slot_until_it_can_enter_working():
    scheduler = _make_scheduler()
    job = PaperJob(arxiv_id="p1", paper=None, workspace_path="", gpu_budget=1.0)
    queued = PaperJob(arxiv_id="p2", paper=None, workspace_path="", gpu_budget=1.0)
    scheduler.queue = deque([queued])
    scheduler.active = {"p1": job}
    scheduler._capacity_state = {"p1": "admitted", "w1": "working", "w2": "working"}
    job.worker_status = "idle"
    job.worker_status_since = 1.0

    t = threading.Thread(target=Scheduler._set_worker_status, args=(scheduler, job, "working"))
    t.start()
    time.sleep(0.1)

    assert job.worker_status == "idle"
    assert job.blocked_on == "working"
    assert scheduler._capacity_state["p1"] == "admitted"
    assert scheduler._launch_calls == 0
    assert queued.status == "queued"

    with scheduler._capacity_cv:
        scheduler._capacity_state.pop("w2", None)
        scheduler._capacity_cv.notify_all()
    t.join(timeout=2)

    assert not t.is_alive()
    assert job.worker_status == "working"
    assert job.blocked_on is None
    assert scheduler._capacity_state["p1"] == "working"
    assert scheduler._launch_calls == 1
    assert queued.status == "active"
    assert scheduler._capacity_state["p2"] == "admitted"


def test_get_status_counts_admitted_separately_from_real_waiting():
    scheduler = _make_scheduler()
    idle_job = PaperJob(arxiv_id="idle", paper=None, workspace_path="", gpu_budget=1.0)
    wait_job = PaperJob(arxiv_id="wait", paper=None, workspace_path="", gpu_budget=1.0)
    build_job = PaperJob(arxiv_id="build", paper=None, workspace_path="", gpu_budget=1.0)
    idle_job.worker_status = "idle"
    wait_job.worker_status = "waiting"
    build_job.worker_status = "building"
    scheduler.active = {
        "idle": idle_job,
        "wait": wait_job,
        "build": build_job,
    }
    scheduler._capacity_state = {
        "idle": "admitted",
        "build": "building",
    }

    status = Scheduler.get_status(scheduler)

    assert status["admitted_count"] == 1
    assert status["waiting_count"] == 1
    assert status["building_count"] == 1


def test_get_status_uses_runtime_bucket_in_legacy_mode():
    scheduler = _make_scheduler()
    scheduler.config = _LegacyConfig()
    idle_job = PaperJob(arxiv_id="idle", paper=None, workspace_path="", gpu_budget=1.0)
    build_job = PaperJob(arxiv_id="build", paper=None, workspace_path="", gpu_budget=1.0)
    install_job = PaperJob(arxiv_id="install", paper=None, workspace_path="", gpu_budget=1.0)
    build_job.worker_status = "building"
    install_job.worker_status = "installing"
    scheduler.active = {
        "idle": idle_job,
        "build": build_job,
        "install": install_job,
    }
    scheduler._capacity_state = {
        "idle": "runtime",
        "build": "runtime",
        "install": "runtime",
    }

    status = Scheduler.get_status(scheduler)

    assert status["split_runtime_caps"] is False
    assert status["admitted_count"] == 0
    assert status["runtime_count"] == 3
    assert status["building_count"] == 1
    assert status["installing_count"] == 1


# ── Excluded papers ───────────────────────────────────────────────────


def test_excluded_papers_default_to_none_when_only_the_example_ships():
    # The repository ships excluded_papers.example.yaml only; the example is
    # a template and must never be applied.
    from swarm import scheduler

    config_dir = scheduler._EXCLUDED_PAPERS_PATH.parent
    assert scheduler._EXCLUDED_PAPERS_PATH.name == "excluded_papers.yaml"
    assert (config_dir / "excluded_papers.example.yaml").is_file()
    if not scheduler._EXCLUDED_PAPERS_PATH.exists():
        assert scheduler._load_excluded_papers() == {}
        scheduler._guard_excluded("0000.00000")  # the example's entry is not excluded


def test_excluded_papers_file_is_applied(monkeypatch, tmp_path: Path):
    import pytest
    from swarm import scheduler

    path = tmp_path / "excluded_papers.yaml"
    path.write_text(
        "excluded:\n"
        "  - arxiv_id: \"1234.56789\"\n"
        "    reason: \"no public artifact\"\n"
        "  - arxiv_id: \"1234.00001\"\n"
    )
    monkeypatch.setattr(scheduler, "_EXCLUDED_PAPERS_PATH", path)

    assert scheduler._load_excluded_papers() == {
        "1234.56789": "no public artifact",
        "1234.00001": "(no reason given)",
    }
    with pytest.raises(ValueError, match="exclude list"):
        scheduler._guard_excluded("1234.56789")
    scheduler._guard_excluded("9999.99999")


def test_excluded_papers_missing_or_malformed_file_excludes_nothing(monkeypatch, tmp_path: Path):
    from swarm import scheduler

    monkeypatch.setattr(scheduler, "_EXCLUDED_PAPERS_PATH", tmp_path / "missing.yaml")
    assert scheduler._load_excluded_papers() == {}

    for content in ("- just\n- a list\n", "excluded: [\n", ""):
        bad = tmp_path / "bad.yaml"
        bad.write_text(content)
        monkeypatch.setattr(scheduler, "_EXCLUDED_PAPERS_PATH", bad)
        assert scheduler._load_excluded_papers() == {}


# ── Nexus auto-start ──────────────────────────────────────────────────


def _scheduler_with_config(**config_kwargs):
    from swarm.config import SwarmConfig

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SwarmConfig(**config_kwargs)
    return scheduler


def test_nexus_autostart_binds_loopback_on_the_nexus_default_port():
    from swarm.config import NexusConnectionConfig

    scheduler = _scheduler_with_config(nexus=NexusConnectionConfig(url="http://localhost"))
    assert scheduler._nexus_autostart_endpoint() == ("127.0.0.1", 39000)

    scheduler = _scheduler_with_config(nexus=NexusConnectionConfig(url="http://localhost:39000"))
    assert scheduler._nexus_autostart_endpoint() == ("127.0.0.1", 39000)


def test_nexus_autostart_follows_url_port_and_configured_host():
    from swarm.config import NexusConnectionConfig

    scheduler = _scheduler_with_config(
        nexus=NexusConnectionConfig(url="http://node01:39555", autostart_host="0.0.0.0"),
    )
    assert scheduler._nexus_autostart_endpoint() == ("0.0.0.0", 39555)


def test_nexus_autostart_script_is_valid_python_bound_to_the_given_host():
    script = Scheduler._nexus_autostart_script("127.0.0.1", 39000)
    compile(script, "<nexus-autostart>", "exec")
    assert "app.run(host='127.0.0.1', port=39000" in script
    assert "0.0.0.0" not in script


def test_restart_nexus_daemon_starts_child_on_loopback(monkeypatch, tmp_path: Path):
    from swarm import scheduler as scheduler_mod
    from swarm.config import NexusConnectionConfig

    scheduler = _scheduler_with_config(nexus=NexusConnectionConfig(url="http://localhost:39000"))
    monkeypatch.setattr(scheduler_mod, "SWARM_DIR", tmp_path)
    monkeypatch.setattr(scheduler_mod.time, "sleep", lambda _s: None)

    run_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append(cmd)
        return scheduler_mod.subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    popen_calls = []

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            popen_calls.append(cmd)

    probes = []

    def fake_urlopen(url, timeout=None):
        probes.append(url)
        return object()

    monkeypatch.setattr(scheduler_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(scheduler_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    scheduler._restart_nexus_daemon()

    assert run_calls == [["lsof", "-ti", "tcp:39000"]]
    assert len(popen_calls) == 1
    script = popen_calls[0][-1]
    assert "host='127.0.0.1', port=39000" in script
    assert probes == ["http://localhost:39000/api/projects"]
    assert scheduler._nexus_proc.pid == 4242
    assert (tmp_path / "nexus.log").exists()


# ── Worker / profile resolution ───────────────────────────────────────


def test_create_worker_supports_both_scaffolds_and_rejects_others():
    import pytest
    from infrastructure.worker.claude_code import ClaudeCodeWorker
    from infrastructure.worker.codex import CodexWorker
    from swarm.config import SUPPORTED_SCAFFOLDS
    from swarm.scheduler import _create_worker

    assert isinstance(_create_worker("claude-code", "m", {}), ClaudeCodeWorker)
    assert isinstance(_create_worker("codex", "m", {}), CodexWorker)
    assert set(SUPPORTED_SCAFFOLDS) == {"claude-code", "codex"}
    for unknown in ("", "claude_code", "terminus-2"):
        with pytest.raises(ValueError, match="Unknown scaffold"):
            _create_worker(unknown, "m", {})


def test_resolve_profile_default_keeps_worker_settings():
    from swarm.config import WorkerConfig

    scheduler = _scheduler_with_config(worker=WorkerConfig(
        scaffold="claude-code", model="m", compact_token_limit=100000,
        api_keys=["k1", "k2"], env_vars={"A": "1"}, container_sif="/sif",
    ))
    wc = scheduler._resolve_profile(None)

    assert (wc.scaffold, wc.model, wc.container_sif) == ("claude-code", "m", "/sif")
    assert wc.compact_token_limit == 100000
    assert wc.api_keys == ["k1", "k2"]
    assert wc.env_vars == {"A": "1"}
    # Copies, not references to the daemon's default config.
    wc.api_keys.append("k3")
    wc.env_vars["B"] = "2"
    assert scheduler.config.worker.api_keys == ["k1", "k2"]
    assert scheduler.config.worker.env_vars == {"A": "1"}


def test_resolve_profile_named_does_not_inherit_default_credentials(tmp_swarm_dir):
    from swarm.config import WorkerConfig

    (tmp_swarm_dir / "config.yaml").write_text(
        "profiles:\n"
        "  other:\n"
        "    scaffold: codex\n"
        "    model: other-model\n"
        "    effort: low\n"
    )
    scheduler = _scheduler_with_config(worker=WorkerConfig(
        scaffold="claude-code", model="m", actual_model="real", max_turns=7,
        compact_token_limit=100000, api_keys=["k1"], env_vars={"TOKEN": "x"},
        container_sif="/sif",
    ))
    wc = scheduler._resolve_profile("other")

    assert (wc.profile, wc.scaffold, wc.model, wc.effort) == ("other", "codex", "other-model", "low")
    # Daemon-wide settings carry over, per-profile credentials do not.
    assert (wc.max_turns, wc.container_sif) == (7, "/sif")
    assert wc.env_vars == {} and wc.api_keys == []
    assert wc.actual_model is None and wc.compact_token_limit is None


def test_resolve_profile_errors_are_clear(tmp_swarm_dir):
    import pytest
    from swarm.config import WorkerConfig

    (tmp_swarm_dir / "config.yaml").write_text("profiles:\n  p:\n    effort: low\n")
    scheduler = _scheduler_with_config(worker=WorkerConfig())

    with pytest.raises(ValueError, match="No scaffold/model configured for the default worker"):
        scheduler._resolve_profile(None)
    with pytest.raises(ValueError, match="No scaffold/model configured for profile 'p'"):
        scheduler._resolve_profile("p")
    with pytest.raises(ValueError, match="Profile 'missing' not found"):
        scheduler._resolve_profile("missing")


# ── Paper listing and recovery ────────────────────────────────────────


def test_get_papers_lists_finished_paused_and_unsuccessful_papers_from_disk(monkeypatch, tmp_path: Path):
    papers_dir = tmp_path / "papers"
    for arxiv_id, status in [
        ("p-done", "done"), ("p-error", "error"), ("p-cancelled", "cancelled"),
        ("p-stalled", "stalled"), ("p-unsuccessful", "unsuccessful"),
        ("p-paused", "paused"), ("p-queued", "queued"),
    ]:
        (papers_dir / arxiv_id).mkdir(parents=True)
        (papers_dir / arxiv_id / "meta.json").write_text(
            json.dumps({"arxiv_id": arxiv_id, "status": status})
        )
    monkeypatch.setattr("swarm.scheduler.PAPERS_DIR", papers_dir)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._lock = threading.Lock()
    scheduler.queue = deque()
    scheduler.active = {}
    scheduler.completed = {}

    listed = {p["arxiv_id"]: p["phase"] for p in scheduler.get_papers()}

    assert listed == {
        "p-done": "completed", "p-error": "completed", "p-cancelled": "completed",
        "p-stalled": "completed", "p-unsuccessful": "completed", "p-paused": "completed",
    }


def test_recover_paper_requires_a_recorded_recovery_action(monkeypatch, tmp_path: Path):
    import pytest

    papers_dir = tmp_path / "papers"
    (papers_dir / "a1").mkdir(parents=True)
    (papers_dir / "a1" / "meta.json").write_text(json.dumps({"arxiv_id": "a1", "status": "done"}))
    monkeypatch.setattr("swarm.scheduler.PAPERS_DIR", papers_dir)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._lock = threading.Lock()
    scheduler.queue = deque()
    scheduler.active = {}
    scheduler.completed = {}

    with pytest.raises(ValueError, match="no recorded recovery action"):
        scheduler.recover_paper("a1")
    with pytest.raises(FileNotFoundError):
        scheduler.recover_paper("unknown")


def test_recover_paper_dispatches_on_recovery_action(monkeypatch, tmp_path: Path):
    papers_dir = tmp_path / "papers"
    (papers_dir / "a1").mkdir(parents=True)
    (papers_dir / "a1" / "meta.json").write_text(json.dumps({
        "arxiv_id": "a1", "status": "paused", "recovery_action": "continue",
    }))
    monkeypatch.setattr("swarm.scheduler.PAPERS_DIR", papers_dir)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._lock = threading.Lock()
    scheduler.queue = deque()
    scheduler.active = {}
    scheduler.completed = {"a1": object()}
    calls = []
    scheduler.continue_paper = lambda arxiv_id, **kwargs: calls.append((arxiv_id, kwargs)) or "job"

    assert scheduler.recover_paper("a1", profile="p", dispatch=False, skip_github=True) == "job"
    assert calls == [("a1", {
        "profile": "p", "no_workflow": False, "fresh": False,
        "dispatch": False, "skip_github": True,
    })]
    assert "a1" not in scheduler.completed


# ── Git and GitHub helpers ────────────────────────────────────────────


def test_required_gitignore_covers_credentials_and_worker_scaffolding():
    from swarm.scheduler import _REQUIRED_GITIGNORE
    from swarm.workspace import WORKSPACE_TEMPLATE_DIR

    for entry in (".git-credentials", ".claude/", ".codex/", ".git-shims/", "trajectory/"):
        assert entry in _REQUIRED_GITIGNORE, entry

    template = (WORKSPACE_TEMPLATE_DIR / ".gitignore").read_text().splitlines()
    for entry in (".git-credentials", ".claude/", ".codex/", ".git-shims/",
                  ".claude.json", ".kimi/", ".gemini/", ".eval/"):
        assert entry in template, entry


def test_ensure_gitignore_adds_missing_required_entries(tmp_path: Path):
    from swarm.scheduler import _REQUIRED_GITIGNORE, _ensure_gitignore

    (tmp_path / ".gitignore").write_text("# agent rewrote this file\n*.pt\n")
    _ensure_gitignore(tmp_path)
    lines = (tmp_path / ".gitignore").read_text().splitlines()

    assert "*.pt" in lines
    assert all(entry in lines for entry in _REQUIRED_GITIGNORE)

    before = (tmp_path / ".gitignore").read_text()
    _ensure_gitignore(tmp_path)  # idempotent
    assert (tmp_path / ".gitignore").read_text() == before


def test_git_env_uses_the_swarm_identity():
    from swarm.scheduler import _git_env

    env = _git_env()
    assert env["GIT_AUTHOR_NAME"] == env["GIT_COMMITTER_NAME"] == "swarm"
    assert env["GIT_AUTHOR_EMAIL"] == env["GIT_COMMITTER_EMAIL"] == "swarm@reproducer"


def test_overview_domain_topics(tmp_path: Path):
    from swarm.scheduler import _overview_domain_topics

    assert _overview_domain_topics(tmp_path) == []
    (tmp_path / "briefing").mkdir()
    (tmp_path / "briefing" / "overview.md").write_text(
        "# Overview\n"
        "- **Domain:** Computer Vision / Self-Supervised Learning, "
        + "x" * 40 + "\n"
        "- **Domain:** ignored second line\n"
    )
    # Slugs longer than 35 characters are dropped.
    assert _overview_domain_topics(tmp_path) == ["computer-vision", "self-supervised-learning"]


def test_filter_repo_is_optional(monkeypatch, tmp_path: Path):
    from swarm import scheduler

    monkeypatch.setattr(scheduler.shutil, "which", lambda name: None)
    called = []
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: called.append(a))

    assert scheduler._filter_repo_strip_paths(tmp_path, ["big.bin"]) is False
    assert scheduler._filter_repo_strip_paths(tmp_path, []) is False
    assert called == []


def test_scan_distinfo_packages_without_any_sif_walks_the_workspace(monkeypatch, tmp_path: Path):
    from swarm import scheduler

    dist = tmp_path / "method" / "vendored" / "demo_pkg-1.2.3.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: Demo-Pkg\nVersion: 1.2.3\n")
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "overlay.img").write_bytes(b"")

    def no_subprocess(*args, **kwargs):
        raise AssertionError("apptainer must not be called without a SIF")

    monkeypatch.setattr(scheduler.subprocess, "run", no_subprocess)

    assert scheduler._scan_distinfo_packages(tmp_path) == {"demo_pkg": "1.2.3"}
    missing = tmp_path / "no-such.sif"
    assert scheduler._scan_distinfo_packages(tmp_path, fallback_sif=str(missing)) == {"demo_pkg": "1.2.3"}


def test_scan_distinfo_packages_mounts_overlay_on_the_fallback_sif(monkeypatch, tmp_path: Path):
    from swarm import scheduler

    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "overlay.img").write_bytes(b"")
    agent_sif = tmp_path / "agent.sif"
    agent_sif.write_bytes(b"")
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return scheduler.subprocess.CompletedProcess(
            cmd, 0, stdout="===DISTINFO===\nName: torch\nVersion: 2.4.0\n", stderr="",
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)

    assert scheduler._scan_distinfo_packages(tmp_path, fallback_sif=str(agent_sif)) == {"torch": "2.4.0"}
    assert len(commands) == 1
    assert commands[0][:2] == ["apptainer", "exec"]
    assert str(agent_sif) in commands[0]


def test_fill_manifest_counts_each_workspace_job_once_across_runs(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace

    from swarm import scheduler

    papers_dir = tmp_path / "papers"
    monkeypatch.setattr(scheduler, "PAPERS_DIR", papers_dir)
    workspace = tmp_path / "ws"
    jobs_dir = workspace / "jobs"

    def add_job(run_id: str) -> None:
        (jobs_dir / run_id).mkdir(parents=True)
        (jobs_dir / run_id / "meta.json").write_text(json.dumps({
            "run_id": run_id, "status": "completed", "num_cpus": 4,
            "started_at": 1000.0, "completed_at": 4600.0,
        }))

    job = PaperJob(
        arxiv_id="p1", paper=SimpleNamespace(arxiv_id="p1", title="T"),
        workspace_path=str(workspace), gpu_budget=1.0,
    )
    wc = SimpleNamespace(scaffold="claude", model="m")

    add_job("r1")
    scheduler._fill_manifest(workspace / "manifest.json", job=job, worker_config=wc)
    add_job("r2")  # second run submits one more job; r1 is still on disk
    scheduler._fill_manifest(workspace / "manifest.json", job=job, worker_config=wc)

    manifest = json.loads((workspace / "manifest.json").read_text())
    assert [[j["run_id"] for j in a["gpu_jobs"]] for a in manifest["agents"]] == [["r1"], ["r2"]]
    assert manifest["reproduction"]["total_gpu_jobs"] == 2
    assert manifest["reproduction"]["total_cpu_hours_used"] == 8.0
