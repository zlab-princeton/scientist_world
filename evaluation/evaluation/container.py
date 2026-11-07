"""Paper container + overlay build helpers.

Each paper environment ships ``environment/container.def`` (the compute
image) and optionally ``environment/setup.sh`` (extra dependencies):

- ``ensure_container(env_dir)`` builds ``environment/container.sif``
  from ``environment/container.def`` synchronously on the current host.
  If a shared ``container_cache_dir`` is configured, the definition is
  hashed (``sha256`` of its bytes, the same key Swarm uses) and a cached
  image is copied instead of rebuilding.
- ``ensure_overlay(env_dir, container_sif)`` creates
  ``environment/overlay.img`` (ext3) and runs ``environment/setup.sh``
  inside the paper container with that overlay mounted. The image is
  built under ``local_tmp_dir`` (local disk, to keep the many small
  writes off a network filesystem) and then copied into the workspace.
  When user namespaces are disabled on the host, a rootless variant is
  used instead (``_ensure_overlay_rootless``).

Both functions are idempotent: if the artifact already exists, they
return immediately. Callers delete the artifact when they need a
rebuild.

There are no file locks and no retries: if two builds race, the first
one wins and the other either hits the cache or fails and is retried
by its caller.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


# Apparent size of a dependency overlay. The file is sparse, so it only
# occupies what setup.sh actually writes.
OVERLAY_SIZE_GB = 30
HASH_FILE = ".build_hashes.json"

# Setup-overlay cache budget (allocated size). Entries are rebuildable
# by construction (a miss just re-runs setup.sh), so evicting is always
# safe; without a cap the cache grows without bound.
OVERLAY_CACHE_BUDGET_GB = 2048
# Never evict an entry this young: an in-flight run may be about to
# copy it, and a half-deleted read is worse than a stale cache.
OVERLAY_CACHE_MIN_AGE_SEC = 2 * 86400


def _hash_file(path: Path) -> str:
    """SHA-256 hex digest of a file (same cache key as Swarm's)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> str | None:
    return _hash_file(path) if path.exists() else None


def _load_hashes(env_dir: Path) -> dict:
    p = env_dir / "environment" / HASH_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_hashes(env_dir: Path, hashes: dict) -> None:
    p = env_dir / "environment" / HASH_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hashes, indent=2))


def snapshot_hashes(env_dir: Path) -> None:
    """Persist hashes of the current container.def + setup.sh.

    Call this once the initial scaffold build is complete so that
    :func:`rebuild_if_changed` doesn't falsely detect the first turn as
    a change.
    """
    env_subdir = env_dir / "environment"
    hashes = {
        "container_def": _hash_path(env_subdir / "container.def"),
        "setup_sh": _hash_path(env_subdir / "setup.sh"),
    }
    _save_hashes(env_dir, hashes)


def _copy_artifact(src: Path, dst: Path) -> None:
    """Copy preserving holes.

    Overlays are created with ``truncate``: sparse files whose allocated
    size is only what ext3 actually wrote (typically a few GB).
    ``shutil.copy2`` reads holes back as zeros and writes them out for
    real, inflating every copy to the full apparent size. GNU ``cp``
    with ``--sparse=always`` reproduces the holes instead; the resulting
    bytes are identical, only the allocation differs.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    rc = 1
    try:
        rc = subprocess.run(
            ["cp", "-p", "--reflink=auto", "--sparse=always",
             str(src), str(tmp)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=3600,
        ).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("sparse cp failed (%s); falling back to copy2", exc)
    if rc != 0:
        # Fall back rather than fail the build — a fat copy still works.
        shutil.copy2(src, tmp)
    tmp.replace(dst)


def _run(cmd: list[str], *, log_path: Path, timeout: int) -> int:
    """Run ``cmd`` and stream stdout+stderr to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(f"\n=== CMD: {' '.join(cmd)} ===\n")
        fh.flush()
        result = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
        )
        fh.write(f"=== RETURN CODE: {result.returncode} ===\n")
    return result.returncode


# ── Container build ──────────────────────────────────────────────────────


def ensure_container(
    env_dir: Path,
    *,
    container_cache_dir: str = "",
    build_timeout: int = 3600,
) -> Path:
    """Ensure ``env_dir/environment/container.sif`` exists.

    Returns the SIF path. Raises ``RuntimeError`` if build fails.
    """
    env_subdir = env_dir / "environment"
    container_sif = env_subdir / "container.sif"
    container_def = env_subdir / "container.def"

    if container_sif.exists():
        return container_sif
    if not container_def.exists():
        raise RuntimeError(
            f"Cannot build container: {container_def} does not exist"
        )

    def_hash = _hash_file(container_def)
    log_path = env_dir / "logs" / "container_build.log"

    # Cache lookup: <cache>/<sha256 of container.def>.sif.
    if container_cache_dir:
        cache_dir = Path(container_cache_dir)
        cached = cache_dir / f"{def_hash}.sif"
        if cached.exists() and cached.stat().st_size > 0:
            log.info("Container cache hit: %s → %s", cached, container_sif)
            _copy_artifact(cached, container_sif)
            return container_sif

    log.info("Building container %s from %s", container_sif, container_def)
    env_subdir.mkdir(parents=True, exist_ok=True)
    rc = _run(
        ["apptainer", "build", "--force", str(container_sif), str(container_def)],
        log_path=log_path, timeout=build_timeout,
    )
    if rc != 0:
        raise RuntimeError(
            f"apptainer build failed (rc={rc}); see {log_path}"
        )

    # Save to cache for future runs.
    if container_cache_dir:
        try:
            cache_dir = Path(container_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            _copy_artifact(container_sif, cache_dir / f"{def_hash}.sif")
            log.info("Cached container build as %s.sif", def_hash)
        except Exception as exc:
            log.warning("Failed to cache container build: %s", exc)

    return container_sif


# ── Overlay build ────────────────────────────────────────────────────────


def _setup_cache_dir(container_cache_dir: str) -> Path | None:
    """Overlay cache location: ``<container_cache_dir>/../setup_overlays``."""
    if not container_cache_dir:
        return None
    return Path(container_cache_dir).parent / "setup_overlays"


def _setup_key(container_hash: str, setup_hash: str) -> str:
    return hashlib.sha256(f"{container_hash}:{setup_hash}".encode()).hexdigest()


def _evict_overlay_cache(cache_dir: Path) -> None:
    """Drop oldest cache entries until under ``OVERLAY_CACHE_BUDGET_GB``.

    Sizes use allocated blocks, not apparent size — a sparse 30 GB image
    holding 2 GB of data costs 2 GB of quota, and evicting on apparent
    size would throw away entries that cost almost nothing.

    Entries are the overlay images and, on hosts without user namespaces,
    the package archives (``<key>.pkgs.tar``). Both are rebuildable (a miss
    re-runs setup.sh), so eviction is always safe. Anything younger than ``OVERLAY_CACHE_MIN_AGE_SEC`` is
    left alone in case an in-flight run is about to copy it.
    """
    budget = OVERLAY_CACHE_BUDGET_GB * (1024 ** 3)
    now = time.time()
    entries = []
    total = 0
    try:
        for p in [*cache_dir.glob("*.img"), *cache_dir.glob("*.pkgs.tar")]:
            try:
                st = p.stat()
            except OSError:
                continue
            # Allocated blocks, except on filesystems that don't report
            # them (st_blocks == 0), where apparent size is the only
            # signal available. Without this fallback the budget check
            # would see a total of 0 and never evict anything.
            size = st.st_blocks * 512 or st.st_size
            total += size
            if (now - st.st_mtime) >= OVERLAY_CACHE_MIN_AGE_SEC:
                entries.append((st.st_mtime, size, p))
    except OSError as exc:
        log.warning("Overlay cache eviction scan failed: %s", exc)
        return
    if total <= budget:
        return
    entries.sort()  # oldest first
    freed = 0
    for _mtime, size, p in entries:
        if total - freed <= budget:
            break
        try:
            p.unlink()
            sidecar = p.with_suffix(".json")
            if sidecar.exists():
                sidecar.unlink()
            freed += size
        except OSError:
            continue
    if freed:
        log.info(
            "Overlay cache eviction: freed %.1f GB (was %.1f GB, budget %d GB)",
            freed / 1024 ** 3, total / 1024 ** 3, OVERLAY_CACHE_BUDGET_GB,
        )


def ensure_overlay(
    env_dir: Path,
    container_sif: Path,
    *,
    container_cache_dir: str = "",
    local_tmp_dir: str = "/tmp",
    uv_cache_dir: str = "",
    setup_timeout: int = 3600,
) -> Path | None:
    """Ensure ``env_dir/environment/overlay.img`` exists, with setup.sh applied.

    Returns the overlay path, or ``None`` if there's no setup.sh (the
    overlay layer is optional — caller should pass ``overlay_path=""``
    to Nexus in that case). Raises ``RuntimeError`` on setup failure.
    """
    env_subdir = env_dir / "environment"
    overlay_img = env_subdir / "overlay.img"
    setup_sh = env_subdir / "setup.sh"

    if overlay_img.exists():
        return overlay_img
    if not setup_sh.exists():
        return None

    setup_hash = _hash_file(setup_sh)
    container_hash = _hash_file(container_sif)
    cache_key = _setup_key(container_hash, setup_hash)
    log_path = env_dir / "logs" / "setup.log"

    # Cache lookup.
    cache_dir = _setup_cache_dir(container_cache_dir)
    if cache_dir is not None:
        cached = cache_dir / f"{cache_key}.img"
        if cached.exists() and cached.stat().st_size > 0:
            log.info("Overlay cache HIT %s → %s", cache_key[:16], overlay_img)
            _copy_artifact(cached, overlay_img)
            return overlay_img
        # Logged so the cache hit rate can be measured.
        log.info(
            "Overlay cache MISS %s (container=%s setup=%s)",
            cache_key[:16], container_hash[:12], setup_hash[:12],
        )

    # Rootless path: on hosts with user namespaces disabled
    # (user.max_user_namespaces=0) the ``apptainer exec --fakeroot``
    # overlay build below cannot run. The rootless build needs no
    # privileges: it binds user-owned staging directories over the
    # container's site-packages/bin paths during setup (so unmodified
    # setup.sh scripts write into them), and a ``usercustomize.py`` in
    # the workspace puts those directories on sys.path at runtime. See
    # _ensure_overlay_rootless.
    if not _userns_available():
        return _ensure_overlay_rootless(
            env_dir, container_sif,
            cache_dir=cache_dir, cache_key=cache_key,
            local_tmp_dir=local_tmp_dir, uv_cache_dir=uv_cache_dir,
            setup_timeout=setup_timeout, log_path=log_path,
        )

    log.info("Creating overlay %s and running setup.sh", overlay_img)
    env_subdir.mkdir(parents=True, exist_ok=True)

    # Build on local disk: setup.sh writes through fuse2fs, which is slow
    # and fragile on a network filesystem.
    local_dir = Path(local_tmp_dir) / "evaluation_overlays"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_overlay = local_dir / f"overlay_{os.getpid()}_{threading.get_ident()}.img"

    try:
        rc = _run(
            ["truncate", "-s", f"{OVERLAY_SIZE_GB}G", str(local_overlay)],
            log_path=log_path, timeout=30,
        )
        if rc != 0:
            raise RuntimeError(f"truncate failed (rc={rc}); see {log_path}")

        rc = _run(
            ["mkfs.ext3", "-q", str(local_overlay)],
            log_path=log_path, timeout=120,
        )
        if rc != 0:
            raise RuntimeError(f"mkfs.ext3 failed (rc={rc}); see {log_path}")

        # Resolve symlinks — fakeroot can't follow host symlink chains.
        real_overlay = str(local_overlay.resolve())
        real_sif = str(container_sif.resolve())
        real_setup = str(setup_sh.resolve())

        exec_cmd = [
            "apptainer", "exec", "--cleanenv", "--fakeroot",
            "--overlay", real_overlay,
        ]
        if uv_cache_dir:
            uv_real = str(Path(uv_cache_dir).resolve())
            Path(uv_real).mkdir(parents=True, exist_ok=True)
            exec_cmd += [
                "--bind", f"{uv_real}:{uv_real}",
                "--env", f"UV_CACHE_DIR={uv_real}",
            ]
        # Bind the host's `uv` binary into the container so `uv pip
        # install ...` works whether or not the paper's container.def
        # installs uv. The example in `prompts/environment.md` uses
        # `uv pip` and agents copy it; without the bind, setup.sh fails
        # with "uv: command not found" on base images without uv. Bind
        # at /usr/bin/uv, not /usr/local/bin/uv: uv installs console
        # scripts into /usr/local/bin under --system, and a single-file
        # bind there makes that directory unwritable for uv.
        host_uv = shutil.which("uv")
        if host_uv:
            host_uv_real = str(Path(host_uv).resolve())
            exec_cmd += ["--bind", f"{host_uv_real}:/usr/bin/uv"]
        exec_cmd += [real_sif, "bash", real_setup]

        rc = _run(exec_cmd, log_path=log_path, timeout=setup_timeout)
        if rc != 0:
            raise RuntimeError(f"setup.sh failed (rc={rc}); see {log_path}")

        _copy_artifact(local_overlay, overlay_img)
    finally:
        if local_overlay.exists():
            try:
                local_overlay.unlink()
            except OSError:
                pass

    # Save to cache for future runs.
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            _copy_artifact(overlay_img, cache_dir / f"{cache_key}.img")
            log.info("Cached overlay as %s.img", cache_key[:16])
            _evict_overlay_cache(cache_dir)
        except Exception as exc:
            log.warning("Failed to cache overlay: %s", exc)

    return overlay_img


# ── Rootless overlay build (no fakeroot / user namespaces needed) ─────────


def _userns_available() -> bool:
    """True when apptainer --fakeroot can work (user namespaces enabled)."""
    try:
        return int(Path("/proc/sys/user/max_user_namespaces").read_text()) > 0
    except (OSError, ValueError):
        return False


_USERCUSTOMIZE = """\
# Auto-generated by evaluation/container.py (rootless package layer).
# Python imports this at startup via the user-site mechanism; it exposes
# the packages installed by environment/setup.sh into environment/pkgs/.
import os as _os
import sys as _sys
_base = '/home/user/environment/pkgs'
for _d in sorted(_os.listdir(_base)) if _os.path.isdir(_base) else []:
    if _d.startswith('site'):
        _p = _os.path.join(_base, _d)
        if _os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)
_b = _os.path.join(_base, 'bin')
if _os.path.isdir(_b) and _b not in _os.environ.get('PATH', ''):
    _os.environ['PATH'] = _b + _os.pathsep + _os.environ.get('PATH', '')
"""


def _container_site_info(container_sif: Path, log_path: Path) -> dict:
    """Ask the container's python3 for its version and /usr site dirs."""
    # Include the standard Debian/Ubuntu site paths even when absent
    # from the image: setup scripts write to them by convention (e.g. a
    # .pth into /usr/lib/pythonX.Y/dist-packages), and apptainer creates
    # missing bind mountpoints via the attached writable overlay.
    code = (
        "import site,os,sys,json;"
        "v='%d.%d'%sys.version_info[:2];"
        "c=[p for p in site.getsitepackages() if p.startswith('/usr')];"
        "c+=['/usr/local/lib/python'+v+'/dist-packages',"
        "'/usr/local/lib/python'+v+'/site-packages',"
        "'/usr/lib/python3/dist-packages',"
        "'/usr/lib/python'+v+'/dist-packages',"
        "'/usr/lib/python'+v+'/site-packages'];"
        "seen=[];"
        "[seen.append(p) for p in c if p not in seen];"
        "print(json.dumps({'py':v,'sites':seen[:6],"
        "'existing':[p for p in seen if os.path.isdir(p)],"
        "'has_local_bin':True}))"
    )
    proc = subprocess.run(
        ["apptainer", "exec", "--cleanenv", str(container_sif), "python3", "-c", code],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"container python3 probe failed (rc={proc.returncode}): "
            f"{(proc.stderr or '')[:300]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _write_usercustomize(env_dir: Path, py_ver: str) -> None:
    usite = env_dir / ".local" / "lib" / f"python{py_ver}" / "site-packages"
    usite.mkdir(parents=True, exist_ok=True)
    (usite / "usercustomize.py").write_text(_USERCUSTOMIZE)


def _blank_overlay(env_dir: Path, local_tmp_dir: str, log_path: Path) -> Path:
    """Create an empty ext3 overlay (no privileges needed)."""
    overlay_img = env_dir / "environment" / "overlay.img"
    local_dir = Path(local_tmp_dir) / "evaluation_overlays"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_overlay = local_dir / f"overlay_{os.getpid()}_{threading.get_ident()}.img"
    try:
        rc = _run(["truncate", "-s", f"{OVERLAY_SIZE_GB}G", str(local_overlay)],
                  log_path=log_path, timeout=30)
        if rc != 0:
            raise RuntimeError(f"truncate failed (rc={rc}); see {log_path}")
        rc = _run(["mkfs.ext3", "-q", str(local_overlay)],
                  log_path=log_path, timeout=120)
        if rc != 0:
            raise RuntimeError(f"mkfs.ext3 failed (rc={rc}); see {log_path}")
        _copy_artifact(local_overlay, overlay_img)
    finally:
        if local_overlay.exists():
            try:
                local_overlay.unlink()
            except OSError:
                pass
    return overlay_img


def _ensure_overlay_rootless(
    env_dir: Path,
    container_sif: Path,
    *,
    cache_dir: Path | None,
    cache_key: str,
    local_tmp_dir: str,
    uv_cache_dir: str,
    setup_timeout: int,
    log_path: Path,
) -> Path:
    """Build the package layer without fakeroot/user namespaces.

    Mechanism:
      1. Probe the container for its python version and /usr site dirs.
      2. SEED user-owned staging dirs (``environment/pkgs/siteN``,
         ``environment/pkgs/bin``) with the image's existing contents of
         those paths (reading the image needs no privileges).
      3. Run the UNMODIFIED setup.sh with the staging dirs bind-mounted
         OVER the corresponding container paths — every install flavor
         (``pip --target``, ``uv pip --system``,
         ``--break-system-packages``) lands in the staging dirs.
      4. Write ``usercustomize.py`` into the workspace's ``.local``
         user-site: at runtime the workspace is ``/home/user``, python
         auto-imports it, and it prepends the staging dirs to sys.path
         (and ``pkgs/bin`` to PATH for python-spawned subprocesses).
      5. Create an EMPTY ext3 overlay for runtime writability (the
         mkfs step needs no privileges; only the old fakeroot install
         step did).

    The staging tree is cached as ``<key>.pkgs.tar`` under the same
    setup-overlay cache key as before.
    """
    env_subdir = env_dir / "environment"
    pkgs_dir = env_subdir / "pkgs"
    overlay_img = env_subdir / "overlay.img"
    setup_sh = env_subdir / "setup.sh"
    env_subdir.mkdir(parents=True, exist_ok=True)

    # Cache hit: unpack the staged package tree.
    if cache_dir is not None:
        cached_tar = cache_dir / f"{cache_key}.pkgs.tar"
        if cached_tar.exists() and cached_tar.stat().st_size > 0:
            log.info("Rootless pkgs cache HIT %s → %s", cache_key[:16], pkgs_dir)
            if pkgs_dir.exists():
                shutil.rmtree(pkgs_dir)
            pkgs_dir.mkdir(parents=True)
            rc = _run(["tar", "-xf", str(cached_tar), "-C", str(pkgs_dir)],
                      log_path=log_path, timeout=1800)
            if rc != 0:
                raise RuntimeError(f"pkgs cache untar failed (rc={rc})")
            meta = _read_pkgs_meta(pkgs_dir)
            _write_usercustomize(env_dir, meta.get("py", "3.10"))
            return _blank_overlay(env_dir, local_tmp_dir, log_path)

    log.info("Rootless overlay build (userns unavailable): %s", env_dir.name)
    info = _container_site_info(container_sif, log_path)
    py_ver = info["py"]
    site_paths = info["sites"] or [f"/usr/local/lib/python{py_ver}/dist-packages"]
    bind_map: list[tuple[Path, str]] = []
    for i, sp in enumerate(site_paths):
        bind_map.append((pkgs_dir / f"site{i}", sp))
    if info.get("has_local_bin"):
        bind_map.append((pkgs_dir / "bin", "/usr/local/bin"))
    # A few setup scripts write runtime env exports into
    # /.singularity.d/env (a fakeroot-era trick). Shadow it so those
    # scripts succeed; the PYTHONPATH/PATH effects they intend are
    # provided by usercustomize.py at runtime.
    bind_map.append((pkgs_dir / "singularity_d", "/.singularity.d"))

    if pkgs_dir.exists():
        shutil.rmtree(pkgs_dir)
    real_sif = str(container_sif.resolve())

    # Seed each staging dir with the image's existing contents so the
    # bind-shadowing doesn't hide packages installed by the container
    # build (%post) — after seeding, staged = image content, and
    # setup.sh's installs merge on top.
    existing = set(info.get("existing", []))
    for stage, cpath in bind_map:
        stage.mkdir(parents=True, exist_ok=True)
        # Paths absent from the image (standard-convention mountpoints
        # bound only so setup scripts may write to them) stay empty.
        if (cpath not in existing
                and cpath not in ("/usr/local/bin", "/.singularity.d")):
            continue
        rc = _run(
            ["bash", "-c",
             f"apptainer exec --cleanenv {shlex.quote(real_sif)} "
             f"bash -c {shlex.quote('test -d ' + cpath + ' || exit 42; tar -C ' + cpath + ' -cf - .')} "
             f"| tar -xf - -C {shlex.quote(str(stage))}"],
            log_path=log_path, timeout=3600,
        )
        if rc == 42:
            continue
        if rc != 0:
            raise RuntimeError(
                f"seeding {cpath} failed (rc={rc}); see {log_path}")

    # Blank overlay first — setup.sh gets it as writable scratch for
    # anything outside the shadowed paths (e.g. /tmp downloads).
    local_dir = Path(local_tmp_dir) / "evaluation_overlays"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_overlay = local_dir / f"overlay_{os.getpid()}_{threading.get_ident()}.img"
    try:
        rc = _run(["truncate", "-s", f"{OVERLAY_SIZE_GB}G", str(local_overlay)],
                  log_path=log_path, timeout=30)
        if rc != 0:
            raise RuntimeError(f"truncate failed (rc={rc}); see {log_path}")
        rc = _run(["mkfs.ext3", "-q", str(local_overlay)],
                  log_path=log_path, timeout=120)
        if rc != 0:
            raise RuntimeError(f"mkfs.ext3 failed (rc={rc}); see {log_path}")

        # Bind the workspace at /home/user during setup, matching the
        # runtime layout — scripts write sentinels (~/.setup_ran) and
        # occasionally pre-download into the workspace.
        exec_cmd = ["apptainer", "exec", "--cleanenv",
                    "--home", f"{env_dir.resolve()}:/home/user",
                    "--overlay", str(local_overlay.resolve())]
        for stage, cpath in bind_map:
            exec_cmd += ["--bind", f"{stage.resolve()}:{cpath}"]
        if uv_cache_dir:
            uv_real = str(Path(uv_cache_dir).resolve())
            Path(uv_real).mkdir(parents=True, exist_ok=True)
            exec_cmd += ["--bind", f"{uv_real}:{uv_real}",
                         "--env", f"UV_CACHE_DIR={uv_real}"]
        host_uv = shutil.which("uv")
        if host_uv:
            exec_cmd += ["--bind", f"{Path(host_uv).resolve()}:/usr/bin/uv"]
        exec_cmd += [real_sif, "bash", str(setup_sh.resolve())]

        rc = _run(exec_cmd, log_path=log_path, timeout=setup_timeout)
        if rc != 0:
            raise RuntimeError(f"setup.sh failed (rc={rc}); see {log_path}")

        _copy_artifact(local_overlay, overlay_img)
    finally:
        if local_overlay.exists():
            try:
                local_overlay.unlink()
            except OSError:
                pass

    # Record the bind mapping + python version for cache restores and
    # debugging, then activate the runtime path injection.
    (pkgs_dir / "pkgs_meta.json").write_text(json.dumps({
        "py": py_ver,
        "binds": [[str(s.relative_to(env_dir)), c] for s, c in bind_map],
        "cache_key": cache_key,
    }, indent=2) + "\n")
    _write_usercustomize(env_dir, py_ver)

    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_tar = cache_dir / f".{cache_key}.pkgs.tar.tmp"
            rc = _run(["tar", "-cf", str(tmp_tar), "-C", str(pkgs_dir), "."],
                      log_path=log_path, timeout=3600)
            if rc == 0:
                tmp_tar.rename(cache_dir / f"{cache_key}.pkgs.tar")
                log.info("Cached rootless pkgs as %s.pkgs.tar", cache_key[:16])
            else:
                tmp_tar.unlink(missing_ok=True)
            _evict_overlay_cache(cache_dir)
        except Exception as exc:
            log.warning("Failed to cache rootless pkgs: %s", exc)

    return overlay_img


def _read_pkgs_meta(pkgs_dir: Path) -> dict:
    try:
        return json.loads((pkgs_dir / "pkgs_meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# ── Agent-driven iteration: rebuild on container.def / setup.sh edits ──


def rebuild_if_changed(
    env_dir: Path,
    *,
    container_cache_dir: str = "",
    local_tmp_dir: str = "/tmp",
    uv_cache_dir: str = "",
) -> tuple[Path, Path | None, bool]:
    """If container.def or setup.sh changed, rebuild the affected layers.

    - container.def changed → ``container.sif`` + ``overlay.img``
      invalidated. Both rebuilt from scratch. Overlay follows because it
      was built for the old base image.
    - setup.sh changed (but not container.def) → overlay rebuilt on top
      of the existing container.sif.

    Returns ``(container_sif, overlay_path, did_rebuild)``. Caller is
    expected to push ``sif_path`` / ``overlay_path`` into Nexus via
    ``update_project`` when ``did_rebuild`` is True.
    """
    env_subdir = env_dir / "environment"
    container_def = env_subdir / "container.def"
    setup_sh = env_subdir / "setup.sh"
    container_sif = env_subdir / "container.sif"
    overlay_img = env_subdir / "overlay.img"

    hashes = _load_hashes(env_dir)
    did_rebuild = False

    # container.def changed → nuke sif + overlay, rebuild both.
    if container_def.exists():
        current_def_hash = _hash_file(container_def)
        if current_def_hash != hashes.get("container_def"):
            log.info("container.def changed — rebuilding container.sif + overlay.img")
            for victim in (container_sif, overlay_img):
                if victim.exists():
                    victim.unlink()
            hashes["container_def"] = current_def_hash
            hashes["setup_sh"] = None  # force overlay rebuild below
            did_rebuild = True

    ensure_container(env_dir, container_cache_dir=container_cache_dir)
    if did_rebuild:
        # Record the new container.def now: if setup.sh fails below, the
        # next call must rebuild only the overlay, not this container.
        _save_hashes(env_dir, hashes)

    # setup.sh changed (or forced above) → rebuild overlay only.
    if setup_sh.exists():
        current_setup_hash = _hash_file(setup_sh)
        if current_setup_hash != hashes.get("setup_sh"):
            log.info("setup.sh changed — rebuilding overlay.img")
            if overlay_img.exists():
                overlay_img.unlink()
            hashes["setup_sh"] = current_setup_hash
            did_rebuild = True

    overlay_path = ensure_overlay(
        env_dir, container_sif,
        container_cache_dir=container_cache_dir,
        local_tmp_dir=local_tmp_dir,
        uv_cache_dir=uv_cache_dir,
    )

    _save_hashes(env_dir, hashes)
    return container_sif, overlay_path, did_rebuild
