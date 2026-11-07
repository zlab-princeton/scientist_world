import json
import sys

import download_arxiv_pdfs


def test_limit_does_not_inflate_already_downloaded(monkeypatch, tmp_path, capsys):
    listing = tmp_path / "arxiv_listings" / "2025-06-09"
    listing.mkdir(parents=True)
    (listing / "cs.LG.json").write_text(json.dumps(["a", "b", "c", "d"]))
    (tmp_path / "arxiv_pdfs").mkdir()
    (tmp_path / "arxiv_pdfs" / "a.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(download_arxiv_pdfs, "download_pdf", lambda aid, dest: "ok")
    monkeypatch.setattr(download_arxiv_pdfs.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", [
        "download_arxiv_pdfs.py", "--date", "2025-06-09", "--limit", "1",
        "--data-dir", str(tmp_path)])

    download_arxiv_pdfs.main()
    assert ("Total: 4  |  To download: 1  |  Already downloaded: 1"
            in capsys.readouterr().out)
