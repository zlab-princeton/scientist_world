import socket
import urllib.error

import pytest

import download_arxiv_pdfs


@pytest.mark.parametrize("error", [
    urllib.error.URLError("connection refused"),
    urllib.error.ContentTooShortError("short read", None),
    socket.timeout("timed out"),
])
def test_network_error_is_reported_as_failed(monkeypatch, tmp_path, error):
    def fake_urlretrieve(url, filename):
        raise error

    monkeypatch.setattr(download_arxiv_pdfs.urllib.request, "urlretrieve",
                        fake_urlretrieve)
    dest = tmp_path / "2501.00001.pdf"
    assert download_arxiv_pdfs.download_pdf("2501.00001", dest) == "failed"
    assert list(tmp_path.iterdir()) == []
