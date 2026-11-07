"""Dashboard: page rendering, read-only API, queue validation and binding.

Uses Flask's test client; no server is started, no run is launched (the
orchestrator entry point is stubbed) and no process is signalled.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

from evaluation import config as _config
from evaluation import dashboard, normalization as nz

PAPER = "2501.00001"

REFERENCE = {"experiments": {"exp1": {
    "metrics": {"acc": {"higher_is_better": True}, "loss": {"higher_is_better": False}},
}}}
SOURCE = {"experiments": {"exp1": {"results": {"ours": {"acc": 0.70, "loss": 2.0}}}}}


def _scores(acc: float, loss: float) -> dict:
    return {"experiments": {"exp1": {"results": {"ours": {"acc": acc, "loss": loss}}}}}


@pytest.fixture
def client(monkeypatch):
    """Test client with a fresh run-scan cache."""
    monkeypatch.setattr(dashboard, "_scan_cache", None)
    monkeypatch.setattr(dashboard, "_scan_refresh_in_flight", False)
    monkeypatch.setattr(dashboard, "_scan_first_done", threading.Event())
    return dashboard.app.test_client()


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _run(data_dir: Path, profile: str, scores: dict | None, *, paper: str = PAPER,
         mode: str = "improve", status: str = "completed", verdict: str | None = "verified",
         test: bool = True, started: str = "2026-01-01T00:00:00+00:00") -> Path:
    """A finished run directory with train (and, by default, test) scores."""
    run_dir = data_dir / "envs" / paper / f"{profile}-{mode}-eval"
    _write(run_dir / ".evaluation.json", {
        "run_id": f"{paper}-{profile}-{mode}", "paper_id": paper, "model": profile,
        "mode": mode, "status": status, "state": "done", "started_at": started})
    if scores is not None:
        _write(run_dir / "scoring" / "scores_train.json", scores)
        if test:
            _write(run_dir / "scoring" / "scores_test.json", scores)
    if verdict is not None:
        ver = {"status": verdict, "grader_profile": "grader", "grader_turns": 3,
               "finished_at": "2026-01-02T03:04:05+00:00"}
        if scores is not None:
            ver.update(scores_train=scores, scores_test=scores)
        _write(run_dir / "scoring" / "verification.json", ver)
    return run_dir


@pytest.fixture
def scored_paper(eval_config) -> Path:
    """Three graded improve runs (the cohort of test_normalization's worked
    example, with equal prior weights) plus the coefficient files."""
    data_dir = eval_config(profiles={
        n: {"scaffold": "codex", "model": n} for n in ("m1", "m2", "m3")})
    scoring = data_dir / "envs" / PAPER / "source" / "scoring"
    _write(scoring / "reference.json", REFERENCE)
    _write(scoring / "scores_train.json", SOURCE)
    _write(scoring / "scores_test.json", SOURCE)
    _run(data_dir, "m1", _scores(0.80, 1.5))
    _run(data_dir, "m2", _scores(0.70, 2.5))
    _run(data_dir, "m3", _scores(0.60, 2.0))
    nz.recompute_paper(PAPER)
    return data_dir


# ── Page ─────────────────────────────────────────────────────────────────


def _display_config(html: str) -> dict:
    m = re.search(r"const DASHBOARD_CONFIG = (.*);\n", html)
    assert m, "display config not embedded in the page"
    return json.loads(m.group(1))


def test_index_renders_with_default_config(client, eval_config) -> None:
    eval_config()
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert _display_config(html) == {"paper_groups": {}, "model_order": []}
    assert "__DASHBOARD_CONFIG__" not in html and "{port}" not in html
    assert "38582" in html


def test_index_embeds_groups_and_model_order_safely(client, eval_config) -> None:
    label = "core </script><img src=x>"
    eval_config(port=40000, dashboard={
        "paper_groups": {label: ["2501.00002", "2501.00001"], "broken": "not a list"},
        "model_order": ["m-b", "m-a"],
    })
    html = client.get("/").get_data(as_text=True)
    assert _display_config(html) == {
        "paper_groups": {label: ["2501.00001", "2501.00002"]},
        "model_order": ["m-b", "m-a"],
    }
    # The label cannot close the script element it is embedded in.
    assert html.count("</script>") == dashboard._HTML.count("</script>")
    assert "<img src=x>" not in html
    assert "40000" in html


def test_page_has_no_built_in_paper_or_model_lists() -> None:
    """Curated subsets come from the config, not from the page source."""
    assert not re.search(r"\b\d{4}\.\d{5}\b", dashboard._HTML), "paper id in the page"
    for name in ("TOP10_IDS", "TOP30_IDS", "_TOP15_IDS"):
        assert name not in dashboard._HTML


def test_verification_table_escapes_every_field() -> None:
    """Notes and statuses are written by the grader model; they are shown
    as text, never interpreted as HTML."""
    m = re.search(r"function renderVerification\(\) \{(.*?)\n\}\n", dashboard._HTML, re.S)
    assert m, "renderVerification not found"
    body = m.group(1)
    exprs = re.findall(r"\$\{([^}]*)\}", body)
    assert exprs
    for expr in exprs:
        if expr == "col":  # looked up in a fixed colour table
            continue
        assert expr.startswith("escapeHtml("), f"unescaped ${{{expr}}}"
    assert "function escapeHtml" in dashboard._HTML


# ── Read-only API ────────────────────────────────────────────────────────


def test_api_options_lists_profiles_and_scaffolded_papers(client, eval_config) -> None:
    data_dir = eval_config(profiles={
        "b": {"scaffold": "codex", "model": "x"}, "a": {"scaffold": "codex", "model": "y"}})
    (data_dir / "envs" / "2501.00002").mkdir(parents=True)
    (data_dir / "envs" / "2501.00001").mkdir(parents=True)
    body = client.get("/api/options").get_json()
    assert body == {"papers": ["2501.00001", "2501.00002"], "models": ["a", "b"],
                    "modes": ["innovate", "improve"]}


def test_api_options_with_a_cohort_file(client, eval_config) -> None:
    data_dir = eval_config(cohort_file="cohort.json")
    (data_dir / "envs" / "2501.00009").mkdir(parents=True)
    # missing file: no filter
    assert client.get("/api/options").get_json()["papers"] == ["2501.00009"]
    (data_dir / "cohort.json").write_text(json.dumps(["2501.00002", "2501.00001"]))
    assert client.get("/api/options").get_json()["papers"] == ["2501.00001", "2501.00002"]
    (data_dir / "cohort.json").write_text(json.dumps({"not": "a list"}))
    assert client.get("/api/options").get_json()["papers"] == ["2501.00009"]


def test_api_runs_and_leaderboard(client, scored_paper: Path) -> None:
    runs = client.get("/api/runs").get_json()
    assert sorted(r["model"] for r in runs) == ["m1", "m2", "m3"]
    m1 = next(r for r in runs if r["model"] == "m1")
    assert m1["status"] == "completed"
    assert m1["train_norm"]["delta_norm"] == pytest.approx(1.5)
    assert "train_score" not in m1, "list view must not carry raw score blobs"

    lb = client.get("/api/leaderboard?mode=improve").get_json()
    assert set(lb) == {"m1", "m2", "m3"}
    # m1 leads on both cells by one spread each: 1 / paper_baseline = 1.5
    assert lb["m1"]["train"]["mean"] == pytest.approx(1.5)
    assert lb["m1"]["test"]["mean"] == pytest.approx(1.5)
    assert lb["m1"]["final"] == {
        "count": 1, "mean": pytest.approx(1.5), "stddev": 0.0,
        "min": pytest.approx(1.5), "max": pytest.approx(1.5)}
    mean_abs = sum(abs(lb[m]["final"]["mean"]) for m in lb) / 3
    assert mean_abs == pytest.approx(1.0)
    assert client.get("/api/leaderboard?mode=innovate").get_json() == {}

    full = client.get(f"/api/run/{PAPER}-m1-improve").get_json()
    assert full["train_score"] == _scores(0.80, 1.5)
    assert client.get("/api/run/nope").status_code == 404


def test_completed_run_without_test_score_is_not_on_the_leaderboard(
    client, scored_paper: Path,
) -> None:
    _run(scored_paper, "m4", _scores(0.9, 1.0), verdict=None, test=False)
    runs = client.get("/api/runs").get_json()
    m4 = next(r for r in runs if r["model"] == "m4")
    assert m4["status"] == "aborted"
    assert "scores_test.json missing" in m4["error"]
    assert "m4" not in client.get("/api/leaderboard?mode=improve").get_json()


def test_final_score_is_the_mean_of_train_and_test() -> None:
    def run(model, train, test, status="completed", mode="improve"):
        return {"model": model, "mode": mode, "status": status,
                "train_norm": {"delta_norm": train}, "test_norm": {"delta_norm": test}}

    runs = [
        run("a", 1.0, 3.0), run("a", None, 5.0), run("a", 7.0, None),
        run("a", float("nan"), float("nan")),
        run("b", 9.0, 9.0, status="incomplete"), run("b", 9.0, 9.0, mode="innovate"),
    ]
    lb = dashboard._aggregate_leaderboard(runs, mode="improve")
    assert set(lb) == {"a"}  # incomplete runs and the other mode are left out
    final = lb["a"]["final"]
    assert final["count"] == 3  # the non-finite score is dropped
    assert final["mean"] == pytest.approx((2.0 + 5.0 + 7.0) / 3)
    assert lb["a"]["train"]["count"] == 2 and lb["a"]["test"]["count"] == 2


def test_comparison_treats_non_finite_agent_values_as_missing() -> None:
    source = {"experiments": {"exp1": {"results": {"ours": {"acc": 0.5, "f1": 0.5}}}}}
    agent = {"experiments": {"exp1": {"results": {"ours": {
        "acc": float("nan"), "f1": 0.4}}}}}
    cmp = dashboard._compute_comparison(agent, source, {})
    assert [c["metric"] for c in cmp["missing"]] == ["acc"]
    assert cmp["n_missing"] == 1 and cmp["n_worse"] == 1
    assert cmp["n_better"] + cmp["n_worse"] + cmp["n_same"] + cmp["n_missing"] == cmp["n_total"]
    assert cmp["delta"] == pytest.approx(-0.2)


def test_leaderboard_chart_items_restricted_to_a_paper_group() -> None:
    def run(paper, model, score):
        return {"paper_id": paper, "model": model, "mode": "improve", "status": "completed",
                "train_norm": {"delta_norm": score}, "test_norm": {"delta_norm": score}}

    runs = [run("p1", "a", 1.0), run("p2", "a", 3.0), run("p1", "b", 2.0)]
    assert dashboard._leaderboard_chart_items(runs, "improve") == [
        ("a", 2.0, 1.0), ("b", 2.0, 0.0)]
    assert dashboard._leaderboard_chart_items(runs, "improve", {"p1"}) == [
        ("b", 2.0, 0.0), ("a", 1.0, 0.0)]
    assert dashboard._leaderboard_chart_items(runs, "improve", set()) == []
    assert dashboard._leaderboard_chart_items(runs, "innovate") == []


def test_chart_for_an_unknown_group_is_a_404(client, eval_config) -> None:
    eval_config(dashboard={"paper_groups": {"core": [PAPER]}})
    resp = client.get("/api/leaderboard.png?mode=improve&group=other")
    assert resp.status_code == 404
    assert "unknown paper group" in resp.get_json()["error"]


def test_chart_renders_a_png(client, scored_paper: Path, eval_config) -> None:
    pytest.importorskip("matplotlib")
    eval_config(dashboard={"paper_groups": {"core": [PAPER]}})
    for url in ("/api/leaderboard.png?mode=improve",
                "/api/leaderboard.png?mode=improve&group=core",
                "/api/leaderboard.png?mode=innovate"):
        resp = client.get(url)
        assert resp.status_code == 200 and resp.mimetype == "image/png"
        assert resp.get_data()[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_route_with_a_stub_plotting_backend(
    client, scored_paper: Path, eval_config, monkeypatch,
) -> None:
    """The route's own logic (group lookup, ordering, x100 scale), with
    matplotlib replaced so the test also runs where it is not installed."""
    eval_config(dashboard={"paper_groups": {"core": [PAPER], "empty": []}})
    fig, ax = mock.MagicMock(), mock.MagicMock()
    plt = mock.MagicMock()
    plt.subplots.return_value = (fig, ax)
    mpl = mock.MagicMock()
    mpl.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)

    resp = client.get("/api/leaderboard.png?mode=improve&group=core")
    assert resp.status_code == 200 and resp.mimetype == "image/png"
    mpl.use.assert_called_once_with("Agg")
    (models, means), kwargs = ax.barh.call_args
    # barh draws its first item at the bottom: worst model first
    assert models == ["m2", "m3", "m1"]
    assert means == pytest.approx([-90.0, -60.0, 150.0])  # FNI scale
    assert kwargs["xerr"] == [0.0, 0.0, 0.0]
    ax.set_title.assert_called_once_with("Improve mode — core (1 papers)")

    ax.reset_mock()
    assert client.get("/api/leaderboard.png?mode=improve&group=empty").status_code == 200
    ax.barh.assert_not_called()
    ax.text.assert_called_once()


def test_api_verification(client, eval_config) -> None:
    data_dir = eval_config()
    _run(data_dir, "m1", None, verdict="verified")
    _run(data_dir, "m2", None, verdict="unfixable", mode="innovate")
    _run(data_dir, "m3", None, verdict="audit_clean")
    long_note = "<b>" + "x" * 500
    ver_path = (data_dir / "envs" / PAPER / "m2-innovate-eval" / "scoring"
                / "verification.json")
    ver = json.loads(ver_path.read_text())
    ver["reason"] = long_note
    ver_path.write_text(json.dumps(ver))

    body = client.get("/api/verification").get_json()
    assert body["counts"] == {"verified": 1, "unfixable": 1, "audit_clean": 1}
    rows = body["rows"]
    assert [(r["model"], r["mode"], r["status"]) for r in rows] == [
        ("m2", "innovate", "unfixable"), ("m1", "improve", "verified"),
        ("m3", "improve", "audit_clean")]
    assert rows[0]["note"] == long_note[:200]
    assert rows[1]["source"] == "grader" and rows[1]["turns"] == 3
    assert rows[2]["source"] == "audit"


# ── Queue ────────────────────────────────────────────────────────────────


@pytest.fixture
def queued(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(dashboard.orchestrator, "bench",
                        lambda papers, models, modes=("innovate",): calls.append(
                            (list(papers), list(models), list(modes))))
    return calls


def test_queue_needs_papers_and_models(client, eval_config, queued) -> None:
    eval_config()
    assert client.post("/api/queue", json={"papers": [PAPER]}).status_code == 400
    assert client.post("/api/queue", json={"models": ["m1"]}).status_code == 400
    assert client.post("/api/queue", data="not json").status_code == 400
    assert queued == []


def test_queue_enqueues_the_matrix(client, eval_config, queued) -> None:
    eval_config()
    resp = client.post("/api/queue", json={
        "papers": ["p1", "p2"], "models": ["m1"], "modes": ["innovate", "improve"]})
    assert resp.get_json() == {"queued": 4}
    assert queued == [(["p1", "p2"], ["m1"], ["innovate", "improve"])]
    # default mode
    client.post("/api/queue", json={"papers": ["p1"], "models": ["m1"]})
    assert queued[-1] == (["p1"], ["m1"], ["innovate"])


def test_queue_rejects_papers_outside_the_cohort_file(client, eval_config, queued) -> None:
    data_dir = eval_config(cohort_file="cohort.json")
    (data_dir / "cohort.json").write_text(json.dumps(["p1"]))
    resp = client.post("/api/queue", json={"papers": ["p1", "p9"], "models": ["m1"]})
    assert resp.status_code == 400
    assert resp.get_json()["non_cohort_papers"] == ["p9"]
    assert queued == []
    assert client.post("/api/queue", json={"papers": ["p1"], "models": ["m1"]}).status_code == 200
    assert len(queued) == 1


def test_queue_state_and_removal(client, eval_config, monkeypatch) -> None:
    eval_config(max_concurrent=3, profile_concurrency={"m1": 2})
    orch = dashboard.orchestrator
    monkeypatch.setattr(orch, "_pending", [
        orch.QueuedRun("p1", "m1", "improve"), orch.QueuedRun("p2", "m2", "innovate")])
    monkeypatch.setattr(orch, "_running", {})
    monkeypatch.setattr(orch, "_max_concurrent_override", None)
    monkeypatch.setattr(orch, "_profile_caps_override", None)

    state = client.get("/api/queue").get_json()
    assert state["max_concurrent"] == 3
    assert state["profile_concurrency"] == {"m1": 2}
    assert state["override_active"] is False
    assert [q["paper_id"] for q in state["pending"]] == ["p1", "p2"]
    assert state["running"] == []
    assert "display_scale" not in state

    assert client.delete("/api/queue", json={"models": ["m1"]}).get_json() == {"removed": 1}
    assert [q.paper_id for q in orch._pending] == ["p2"]


def test_concurrency_validation(client, eval_config, monkeypatch) -> None:
    eval_config()
    orch = dashboard.orchestrator
    monkeypatch.setattr(orch, "_max_concurrent_override", None)
    monkeypatch.setattr(orch, "_profile_caps_override", None)
    monkeypatch.setattr(orch, "_executor", None)

    for bad in ({"max_concurrent": "many"}, {"max_concurrent": 0},
                {"profile_concurrency": ["m1"]}, {"profile_concurrency": {"m1": "x"}},
                {"profile_concurrency": {"m1": -1}}):
        assert client.post("/api/concurrency", json=bad).status_code == 400, bad
    assert orch._max_concurrent_override is None

    body = client.post("/api/concurrency", json={
        "max_concurrent": 4, "profile_concurrency": {"m1": 1}}).get_json()
    assert body["max_concurrent"] == 4
    assert body["profile_concurrency"] == {"m1": 1}
    assert body["override_active"] is True


# ── Binding ──────────────────────────────────────────────────────────────


@pytest.fixture
def served(monkeypatch) -> dict:
    """Capture what serve() would bind, without sweeping or serving."""
    seen: dict = {"sweeps": 0}

    def sweep() -> None:
        seen["sweeps"] += 1

    monkeypatch.setattr(dashboard, "_sweep_orphan_agents", sweep)
    monkeypatch.setattr(dashboard, "_sweep_orphan_run_artifacts", sweep)
    monkeypatch.setattr(dashboard.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(dashboard.app, "run",
                        lambda host, port, **k: seen.update(host=host, port=port, kwargs=k))
    return seen


def test_serve_binds_loopback_by_default(eval_config, served) -> None:
    eval_config()
    dashboard.serve()
    assert (served["host"], served["port"]) == ("127.0.0.1", 38582)
    assert served["kwargs"]["debug"] is False


def test_serve_host_and_port_from_config_or_arguments(eval_config, served) -> None:
    eval_config(host="0.0.0.0", port=40001)
    dashboard.serve()
    assert (served["host"], served["port"]) == ("0.0.0.0", 40001)
    dashboard.serve(port=40002, host="192.0.2.1")
    assert (served["host"], served["port"]) == ("192.0.2.1", 40002)


def test_serve_without_data_dir_fails_before_doing_anything(served) -> None:
    with pytest.raises(_config.ConfigError, match="data_dir"):
        dashboard.serve()
    assert served == {"sweeps": 0}
