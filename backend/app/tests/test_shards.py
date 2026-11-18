"""Shard mode with the network replaced: 'downloads' copy fixture images, uploads are faked."""
import base64
import hashlib
import json
import shutil

from typer.testing import CliRunner

from app.cli import app
from app.core.workspace import Workspace
from app.services import hf_upload, shards
from app.services.shards import is_url_list_dir, read_urls

runner = CliRunner()
GOOD = ["scene_b.png", "scene_d.jpg", "scene_c.webp", "nested/scene_f.png", "nested/scene_e_portrait.jpg", "rotated.jpg",
        "with_gps.jpg", "scene_g.jpg", "alpha.png", "cmyk.jpg", "text_image.png", "nested/shapes.tif"]


def make_lists(folder, fixture_dir):
    """Two URL lists pointing at fixture images, plus one dead URL and one repeated URL."""
    folder.mkdir()
    rows, files = [], {}
    for name in GOOD:
        data = (fixture_dir / name).read_bytes()
        url = f"https://example.invalid/{name.replace('/', '_')}"
        files[url] = fixture_dir / name
        rows.append(f"{url}\t{len(data)}\t{base64.b64encode(hashlib.md5(data).digest()).decode()}")
    rows.insert(3, "https://example.invalid/removed.jpg\t1234\tAAAAAAAAAAAAAAAAAAAAAA==")
    rows.append(rows[0])  # the same file listed twice
    (folder / "list0.tsv").write_text("TsvHttpData-1.0\n" + "\n".join(rows[:7]) + "\n")
    (folder / "list1.tsv").write_text("TsvHttpData-1.0\n" + "\n".join(rows[7:]) + "\n")
    return files, len(rows)


def fake_net(monkeypatch, files, uploads):
    def download_one(url, size, md5, dest, cfg, ctx):
        if url not in files:
            return "gone", "HTTP 404", 0
        shutil.copyfile(files[url], dest)
        return "ok", "", size

    def upload(args):
        uploads.append({k: args[k] for k in ("repo_id", "path_in_repo", "private")} | {"files": sorted(
            p.name for p in __import__("pathlib").Path(args["folder"], "images").iterdir()), "readme": args["readme"]})
        return {"ok": True, "repo_url": "https://huggingface.co/datasets/" + args["repo_id"]}

    monkeypatch.setattr(shards, "download_one", download_one)
    monkeypatch.setattr(hf_upload, "_run_child", upload)


def test_url_list_detection_and_cursor(tmp_path, fixture_dir):
    lists = tmp_path / "lists"
    _, n = make_lists(lists, fixture_dir)
    assert is_url_list_dir(lists) and not is_url_list_dir(fixture_dir) and not is_url_list_dir(tmp_path / "nope")
    state = {"source": str(lists), "files": shards.list_files(lists), "file_idx": 0, "offset": 0, "index": 0}
    assert shards.count_urls(lists, state["files"]) == n
    it = read_urls(lists, state)
    first = [next(it) for _ in range(5)]
    assert [r[0] for r in first] == [0, 1, 2, 3, 4]
    # resume from the stored cursor: continues with the next URL, across the file boundary
    cursor = {**state, "file_idx": first[-1][4], "offset": first[-1][5], "index": 5}
    rest = list(read_urls(lists, cursor))
    assert [r[0] for r in rest] == list(range(5, n)) and rest[0][1] != first[-1][1]


def test_shards_are_closed_uploaded_and_cleaned(tmp_path, fixture_dir, monkeypatch):
    lists = tmp_path / "lists"
    files, n = make_lists(lists, fixture_dir)
    uploads = []
    fake_net(monkeypatch, files, uploads)
    root = tmp_path / "ws"
    for k, v in (("HF_DATASET_REPO", "someone/shard-test"), ("HF_MIN_SIZE_MB", "0.5"), ("HF_PRIVATE", "true")):
        monkeypatch.setenv(k, v)
    before = {p.name: p.read_bytes() for p in lists.iterdir()}
    w = ["--workspace", str(root)]

    no_rights = runner.invoke(app, ["run", str(lists), *w])
    assert no_rights.exit_code == 1 and "rights status" in no_rights.output and not uploads  # nothing is fetched or published blindly

    res = runner.invoke(app, ["run", str(lists), "--rights-status", "user_attested", "--workers", "2", "--max-urls", "8", *w])
    assert res.exit_code == 0, res.output
    out = res.output
    assert "shard mode" in out and "[DOWNLOAD] url_000000000 ✓" in out and "gone: HTTP 404" in out
    assert "shard_0001 closed" in out and "shard_0001 uploaded" in out and "local copy removed" in out
    assert uploads and uploads[0]["path_in_repo"] == "shards/shard_0001" and uploads[0]["private"] is True
    assert uploads[0]["files"] and "shard_0001" in uploads[0]["readme"] and "not verified" in uploads[0]["readme"]
    assert not (root / "shards" / "shard_0001").exists() and not (root / "downloads" / "shard_0001").exists()  # disk is freed

    ws = Workspace(root)
    st = shards.ShardRunner(ws, None).load()
    assert st["index"] == 8 and st["gone"] == 1 and st["shards_done"][0]["name"] == "shard_0001"
    with ws.db.session() as s:  # the hashes of what was exported stay behind for later shards to be compared against
        from app.database.models import SeenHash

        seen = list(s.query(SeenHash))
    assert len(seen) == sum(e["records"] for e in st["shards_done"]) > 0 and all(r.shard.startswith("shard_") and r.phash for r in seen)
    assert st["shards_done"][0]["stats"]["categories"] and "configs:" in uploads[0]["readme"] and "Licence notice" in uploads[0]["readme"]
    ws.db.dispose()

    again = runner.invoke(app, ["resume", *w])  # the target of 8 URLs is reached: nothing more to fetch
    assert again.exit_code == 0 and "[DOWNLOAD] url_" not in again.output
    rest = runner.invoke(app, ["run", str(lists), "-q", *w])  # no limit: continue through the lists
    assert rest.exit_code == 0, rest.output
    ws = Workspace(root)
    st = shards.ShardRunner(ws, None).load()
    assert st["index"] == n and st["duplicates"] == 1 and st["downloaded"] == len(GOOD)
    shipped = [f for u in uploads for f in u["files"] if f.startswith("img_")]
    assert len(shipped) == len(set(shipped))  # no image is uploaded twice
    assert [u["path_in_repo"] for u in uploads] == [f"shards/shard_{i + 1:04d}" for i in range(len(uploads))]
    status = runner.invoke(app, ["status", "--workspace", str(root)]).output
    assert "Shard mode" in status and f"{n} / {n}" in status and "shards uploaded" in status
    ws.db.dispose()
    assert {p.name: p.read_bytes() for p in lists.iterdir()} == before  # the URL lists are never modified


def test_failed_upload_keeps_the_shard_and_is_retried(tmp_path, fixture_dir, monkeypatch):
    lists = tmp_path / "lists"
    files, n = make_lists(lists, fixture_dir)
    uploads = []
    fake_net(monkeypatch, files, uploads)
    monkeypatch.setattr(hf_upload, "_run_child", lambda args: {"ok": False, "error": "not logged in to Hugging Face"})
    for k, v in (("HF_DATASET_REPO", "someone/shard-test"), ("HF_MIN_SIZE_MB", "0.5")):
        monkeypatch.setenv(k, v)
    root = tmp_path / "ws"
    res = runner.invoke(app, ["run", str(lists), "--rights-status", "user_attested", "-q", "--max-urls", "8", "--workspace", str(root)])
    assert res.exit_code == 1 and "not logged in" in res.output and "image-pipeline resume" in res.output
    kept = root / "shards" / "shard_0001"
    assert kept.is_dir() and any((kept / "images").glob("img_*"))  # nothing is deleted before the upload succeeded
    fake_net(monkeypatch, files, uploads)
    ok = runner.invoke(app, ["resume", "-q", "--workspace", str(root)])
    assert ok.exit_code == 0, ok.output
    assert uploads[0]["path_in_repo"] == "shards/shard_0001" and not kept.exists()


def test_hugging_face_parquet_source(tmp_path, fixture_dir, monkeypatch):
    """A Hub dataset of parquet files with embedded image bytes is consumed row by row."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from app.services.shards import parse_hf_source

    assert parse_hf_source("https://huggingface.co/datasets/UWGZQ/pixmo_images") == "UWGZQ/pixmo_images"
    assert parse_hf_source("hf://org/name") == "org/name" and parse_hf_source("/some/folder") is None
    assert parse_hf_source("https://huggingface.co/datasets/../etc") is None

    names = GOOD + ["corrupt.jpg"]
    remote = tmp_path / "remote"
    remote.mkdir()
    half = len(names) // 2
    for i, part in enumerate((names[:half], names[half:])):
        rows = [{"image_id": hashlib.sha256(n.encode()).hexdigest(), "ext": "jpg",  # ext column is wrong on purpose for png/webp
                 "image": {"bytes": (fixture_dir / n).read_bytes(), "path": None}} for n in part]
        if i == 1:
            rows.append({"image_id": "gifrow", "ext": "gif", "image": {"bytes": b"GIF89a" + b"\x00" * 50, "path": None}})
            rows.append(dict(rows[0], image_id="copy_of_first"))  # same bytes under another id
        pq.write_table(pa.Table.from_pylist(rows), remote / f"train-0000{i}-of-00002.parquet")
    fetched = []

    def info(repo, ctx):
        return {"files": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"], "rows": len(names) + 2,
                "license": "mixed-web-sources"}

    def fetch(url, dest, ctx, timeout, say):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / url.rsplit("/", 1)[-1], dest)
        return dest.stat().st_size

    monkeypatch.setattr(shards, "hf_dataset_info", info)
    monkeypatch.setattr(shards, "download_file", fetch)
    uploads = []
    monkeypatch.setattr(hf_upload, "_run_child", lambda a: uploads.append(a) or {"ok": True, "repo_url": "https://huggingface.co/datasets/me/out"})
    for k, v in (("HF_DATASET_REPO", "me/out"), ("HF_MIN_SIZE_MB", "0.5")):
        monkeypatch.setenv(k, v)
    root = tmp_path / "ws"
    w = ["--workspace", str(root)]

    first = runner.invoke(app, ["run", "hf://org/name", "--rights-status", "user_attested", "--max-images", "4", *w])
    assert first.exit_code == 0, first.output
    assert "[EXTRACT] row_000000000 ✓" in first.output and len(fetched) == 1
    rest = runner.invoke(app, ["resume", "-q", *w])            # reaches the stored target: nothing new
    assert rest.exit_code == 0 and len(fetched) == 1
    done = runner.invoke(app, ["run", "hf://org/name", "-q", *w])  # no limit: through both files, first file not fetched again
    assert done.exit_code == 0, done.output
    assert [u.rsplit("/", 1)[-1] for u in fetched] == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    assert fetched[0].startswith("https://huggingface.co/datasets/org/name/resolve/main/data/")

    ws = Workspace(root)
    st = shards.ShardRunner(ws, None).load()
    assert st["kind"] == "hf" and st["index"] == len(names) + 2 and st["file_idx"] == 2
    assert st["unsupported"] == 1 and st["duplicates"] == 1 and st["downloaded"] == len(names)
    assert "mixed-web-sources" in st["license"] and "not verified" in st["license"]
    assert not list((root / "downloads" / "_source").glob("*.parquet"))  # consumed source files are removed
    assert uploads and all(u["path_in_repo"].startswith("shards/shard_") for u in uploads) and uploads[0]["repo_id"] == "me/out"
    assert "mixed-web-sources" in uploads[-1]["readme"]
    ws.db.dispose()


def test_background_download_keeps_at_most_three_source_files(tmp_path):
    """While file N is processed, N+1 and N+2 are fetched; N+3 only after N is consumed and deleted."""
    import threading
    import time

    from app.services.shards import Prefetcher

    cache = tmp_path / "src"
    state = {"files": [f"data/f{i}.parquet" for i in range(6)], "file_idx": 0, "repo": "org/name"}
    order, peak = [], [0]
    gate = threading.Event()

    def fetch(url, dest, ctx, timeout, say):
        order.append(url.rsplit("/", 1)[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        peak[0] = max(peak[0], len(list(cache.glob("*.parquet"))))
        return 1

    import app.services.shards as mod
    real = mod.download_file
    mod.download_file = fetch
    try:
        pf = Prefetcher(state, cache, 3, 5, lambda m: None)
        pf.start()
        first = pf.wait_for(0)
        deadline = time.time() + 5
        while len(order) < 3 and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)                                   # give it the chance to (wrongly) fetch a fourth
        assert order == ["f0.parquet", "f1.parquet", "f2.parquet"] and pf.on_disk() == 3
        first.unlink()                                    # file 0 is unpacked ...
        pf.hold = 0                                       # ... and its images are still being processed
        state["file_idx"] = 1
        time.sleep(0.8)
        assert order == ["f0.parquet", "f1.parquet", "f2.parquet"]   # the 4th must wait
        pf.release()                                      # processing of file 0 finished
        deadline = time.time() + 5
        while len(order) < 4 and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        assert order == ["f0.parquet", "f1.parquet", "f2.parquet", "f3.parquet"] and pf.on_disk() == 3
        for i in (1, 2, 3, 4):                            # run through the rest
            pf.wait_for(i).unlink()
            state["file_idx"] = i + 1
        pf.wait_for(5)
        assert order == [f"f{i}.parquet" for i in range(6)] and peak[0] <= 3
        pf.stop()
    finally:
        mod.download_file = real
