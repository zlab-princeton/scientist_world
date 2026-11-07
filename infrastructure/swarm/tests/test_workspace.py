"""Tests for workspace creation."""

import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch

from swarm.paper import PaperInfo
from swarm.workspace import create_workspace, _ensure_git_remote_url, _install_pre_commit_hook


class TestCreateWorkspace:
    """Workspace creation and seeding."""

    def _make_template(self, tmp_path) -> Path:
        """Create a minimal workspace template."""
        template = tmp_path / "template"
        template.mkdir()
        (template / "action.yaml").write_text("action: submit\n")
        (template / "protocol.md").write_text("# Protocol\n")
        (template / "scripts").mkdir()
        (template / "scripts" / "run.sh").write_text("#!/bin/bash\necho hello\n")
        (template / "paper").mkdir()
        return template

    def _make_paper(self, tmp_path) -> PaperInfo:
        """Create a PaperInfo with real files."""
        data = tmp_path / "data"
        md_dir = data / "arxiv_mds" / "2512.01152"
        md_dir.mkdir(parents=True)
        md_file = md_dir / "2512.01152.md"
        md_file.write_text("# My Paper")
        images = md_dir / "images"
        images.mkdir()

        pdf_dir = data / "arxiv_pdfs"
        pdf_dir.mkdir(parents=True)
        pdf_file = pdf_dir / "2512.01152.pdf"
        pdf_file.write_bytes(b"%PDF")

        return PaperInfo(
            arxiv_id="2512.01152",
            title="Test Paper Title",
            paper_md_path=str(md_file),
            paper_images_dir=str(images),
            paper_pdf_path=str(pdf_file),
        )

    @patch("swarm.workspace._setup_git_credentials")
    @patch("swarm.workspace._git_init")
    def test_creates_workspace(self, mock_git, mock_creds, tmp_path):
        """Workspace is created with template files and paper symlinks."""
        template = self._make_template(tmp_path)
        paper = self._make_paper(tmp_path)
        base_dir = tmp_path / "workspaces"
        base_dir.mkdir()

        ws_path = create_workspace(paper, str(base_dir), str(template))

        ws = Path(ws_path)
        assert ws.exists()
        assert (ws / "action.yaml").exists()
        assert (ws / "protocol.md").exists()
        assert (ws / "scripts" / "run.sh").exists()

        # Paper files are copied (not symlinked) for container compatibility
        assert (ws / "paper" / "2512.01152.md").exists()
        assert not (ws / "paper" / "2512.01152.md").is_symlink()
        assert (ws / "paper" / "2512.01152.md").read_text() == "# My Paper"
        assert (ws / "paper" / "images").is_dir()
        assert not (ws / "paper" / "images").is_symlink()
        assert (ws / "paper" / "2512.01152.pdf").exists()
        assert not (ws / "paper" / "2512.01152.pdf").is_symlink()

    @patch("swarm.workspace._setup_git_credentials")
    @patch("swarm.workspace._git_init")
    def test_raises_on_existing_workspace(self, mock_git, mock_creds, tmp_path):
        """Raise FileExistsError if workspace already exists."""
        template = self._make_template(tmp_path)
        paper = self._make_paper(tmp_path)
        base_dir = tmp_path / "workspaces"
        base_dir.mkdir()

        # Create workspace once
        create_workspace(paper, str(base_dir), str(template))

        # Second call should fail
        with pytest.raises(FileExistsError):
            create_workspace(paper, str(base_dir), str(template))

    def test_raises_on_missing_template(self, tmp_path):
        """Raise FileNotFoundError if template doesn't exist."""
        paper = self._make_paper(tmp_path)

        with pytest.raises(FileNotFoundError, match="Workspace template not found"):
            create_workspace(paper, str(tmp_path), str(tmp_path / "nonexistent"))


class TestGitRemoteSetup:
    def test_ensure_git_remote_url_tolerates_timeout_if_remote_was_added(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()

        remote_url = "https://github.com/example/repo.git"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:3] == ["git", "remote", "get-url"]:
                get_url_calls = sum(1 for c in calls if c[:3] == ["git", "remote", "get-url"])
                if get_url_calls == 1:
                    return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="No such remote")
                return subprocess.CompletedProcess(cmd, 0, stdout=remote_url + "\n", stderr="")
            if cmd[:3] == ["git", "remote", "add"]:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))
            raise AssertionError(f"Unexpected command: {cmd}")

        with patch("swarm.workspace.subprocess.run", side_effect=fake_run):
            _ensure_git_remote_url(workspace, "origin", remote_url)

        assert any(cmd[:3] == ["git", "remote", "add"] for cmd in calls)

    def test_ensure_git_remote_url_updates_existing_remote(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()

        remote_url = "https://github.com/example/repo.git"
        old_url = "https://github.com/example/old.git"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:3] == ["git", "remote", "get-url"]:
                get_url_calls = sum(1 for c in calls if c[:3] == ["git", "remote", "get-url"])
                if get_url_calls == 1:
                    return subprocess.CompletedProcess(cmd, 0, stdout=old_url + "\n", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout=remote_url + "\n", stderr="")
            if cmd[:3] == ["git", "remote", "set-url"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            raise AssertionError(f"Unexpected command: {cmd}")

        with patch("swarm.workspace.subprocess.run", side_effect=fake_run):
            _ensure_git_remote_url(workspace, "origin", remote_url)

        assert any(cmd[:3] == ["git", "remote", "set-url"] for cmd in calls)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs the git executable")
class TestPreCommitHook:
    @staticmethod
    def _commit(repo, rel_path):
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                 "-c", f"core.hooksPath={repo / '.git' / 'hooks'}", *args],
                capture_output=True, text=True,
            )
        git("init", "-q")
        _install_pre_commit_hook(repo)
        target = repo / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n")
        git("add", "-f", rel_path)
        return git("commit", "-q", "-m", "add")

    @pytest.mark.parametrize("rel_path", [
        ".venv/lib/mod.py",
        "site-packages/pkg/mod.py",
        "node_modules/pkg/index.js",
        "__pycache__/mod.cpython-311.pyc",
        ".tox/py311/mod.py",
        "src/.venv/lib/mod.py",
    ])
    def test_blocks_installed_content_at_any_depth(self, tmp_path, rel_path):
        result = self._commit(tmp_path, rel_path)
        assert result.returncode != 0, result.stderr
        assert "never-legitimate" in result.stderr

    def test_allows_ordinary_source(self, tmp_path):
        result = self._commit(tmp_path, "src/mod.py")
        assert result.returncode == 0, result.stderr
