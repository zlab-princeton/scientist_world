import json
import sys

import analyze_papers


def test_limit_is_applied_once_across_retry_passes(monkeypatch, tmp_path):
    ids = ["a", "b", "c", "d"]
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(ids))
    for aid in ids:
        (tmp_path / "arxiv_mds" / aid).mkdir(parents=True)
        (tmp_path / "arxiv_pdfs").mkdir(exist_ok=True)
        (tmp_path / "arxiv_pdfs" / f"{aid}.pdf").write_bytes(b"%PDF")
    # main() rebinds these globals for --data-dir; restore them afterwards.
    for name in ("DATA_DIR", "LISTING_DIR", "PDF_DIR", "MD_DIR", "ANALYSIS_DIR"):
        monkeypatch.setattr(analyze_papers, name, getattr(analyze_papers, name))
    attempted = []

    def fake_run_agent(arxiv_id, folder, model_cfg, max_retries):
        attempted.append(arxiv_id)
        if arxiv_id == "b":                     # b never succeeds
            return False
        (folder / analyze_papers.DONE_MARKER).touch()
        return True

    monkeypatch.setattr(analyze_papers, "run_agent", fake_run_agent)
    monkeypatch.setattr(analyze_papers.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", [
        "analyze_papers.py", "--ids-file", str(ids_file), "--limit", "2",
        "--data-dir", str(tmp_path), "--pass-retries", "1"])

    analyze_papers.main()
    assert set(attempted) == {"a", "b"}
