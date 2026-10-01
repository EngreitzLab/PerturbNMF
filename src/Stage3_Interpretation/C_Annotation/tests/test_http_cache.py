"""Shards sharing one cache dir: concurrent saves merge, none overwrites another's entries."""
import io
import json
import multiprocessing
import urllib.request

import http_cache
from http_cache import CachedHttp


def fake_urlopen(request, timeout):
    return io.BytesIO(json.dumps({"url": request.full_url}).encode())


def fetch_and_save(cache_dir, prefix, both_loaded):
    urllib.request.urlopen = fake_urlopen
    http = CachedHttp(cache_dir, pause=0)  # both shards load the cache before either saves
    both_loaded.wait()
    for i in range(20):
        http.get_json(f"https://example.org/{prefix}{i}")
        http.save()


def test_concurrent_saves_keep_both_processes_entries(tmp_path):
    (tmp_path / "http_cache.json").write_text(json.dumps({"old": 1, "failed": None}))
    context = multiprocessing.get_context("spawn")
    both_loaded = context.Barrier(2)
    workers = [context.Process(target=fetch_and_save, args=(tmp_path, prefix, both_loaded)) for prefix in "ab"]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    cache = json.loads((tmp_path / "http_cache.json").read_text())
    assert set(cache) == {"old"} | {f"https://example.org/{p}{i}" for p in "ab" for i in range(20)}
    assert not list(tmp_path.glob("*.tmp"))


def test_save_picks_up_entries_written_by_another_process(tmp_path, monkeypatch):
    monkeypatch.setattr(http_cache.urllib.request, "urlopen", fake_urlopen)
    first, second = CachedHttp(tmp_path, pause=0), CachedHttp(tmp_path, pause=0)
    first.get_json("https://example.org/x")
    first.save()
    second.get_json("https://example.org/y")
    second.save()
    assert set(json.loads((tmp_path / "http_cache.json").read_text())) == {"https://example.org/x", "https://example.org/y"}
    assert second.get_json("https://example.org/x") == {"url": "https://example.org/x"}
