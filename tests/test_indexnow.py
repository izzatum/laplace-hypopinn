"""Boundaries for project-scoped IndexNow notifications; no live requests."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from xml.sax.saxutils import escape

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "submit_indexnow.py"
SITE_URL = "https://izzatum.github.io/laplace-hypopinn/"
KEY = "0123456789abcdef0123456789abcdef"


def write_inputs(tmp_path, urls):
    key_file = tmp_path / f"{KEY}.txt"
    key_file.write_text(KEY, encoding="ascii")
    sitemap = tmp_path / "sitemap.xml"
    entries = "".join(f"<url><loc>{escape(url)}</loc></url>" for url in urls)
    sitemap.write_text(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + entries + "</urlset>", encoding="utf-8"
    )
    return sitemap, key_file


@pytest.fixture
def indexnow():
    spec = importlib.util.spec_from_file_location("submit_indexnow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_outputs_project_payload(tmp_path):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL, SITE_URL + "methods/"])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--sitemap", str(sitemap),
         "--key-file", str(key_file)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "host": "izzatum.github.io", "key": KEY,
        "keyLocation": SITE_URL + key_file.name,
        "urlList": [SITE_URL, SITE_URL + "methods/"],
    }


@pytest.mark.parametrize("url", [
    "https://izzatum.github.io/", "https://izzatum.github.io/another-project/",
    "https://izzatum.github.io/laplace-hypopinn-extra/",
    "https://other.example/laplace-hypopinn/", "http://izzatum.github.io/laplace-hypopinn/",
    SITE_URL + "../another-project/", SITE_URL + "%2e%2e/another-project/",
    SITE_URL + "%252e%252e/another-project/", SITE_URL + "..\\another-project/",
    SITE_URL + "methods/?duplicate=1", SITE_URL + "methods/#heading",
    "https://user@izzatum.github.io/laplace-hypopinn/", SITE_URL + "bad\tpath/",
])
def test_rejects_urls_outside_canonical_project_scope(indexnow, tmp_path, url):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL, url])
    with pytest.raises(ValueError):
        indexnow.build_payload(sitemap, key_file)


@pytest.mark.parametrize("site_url", [
    "http://izzatum.github.io/laplace-hypopinn/", "https://izzatum.github.io/",
    "https://user@izzatum.github.io/laplace-hypopinn/", SITE_URL + "?query=1",
    "https://izzatum.github.io/laplace-hypopinn", SITE_URL + "../",
])
def test_rejects_ambiguous_site_scope(indexnow, tmp_path, site_url):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL])
    with pytest.raises(ValueError):
        indexnow.build_payload(sitemap, key_file, site_url)


@pytest.mark.parametrize("urls", [[], [SITE_URL, SITE_URL]])
def test_rejects_empty_or_duplicate_urls(indexnow, tmp_path, urls):
    sitemap, key_file = write_inputs(tmp_path, urls)
    with pytest.raises(ValueError):
        indexnow.build_payload(sitemap, key_file)


def test_rejects_more_than_protocol_limit(indexnow, tmp_path):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL + f"{i}/" for i in range(10_001)])
    with pytest.raises(ValueError):
        indexnow.build_payload(sitemap, key_file)


@pytest.mark.parametrize("content", ["wrong-key", KEY + "\n", "", "not a valid key"])
def test_requires_key_content_to_match_filename(indexnow, tmp_path, content):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL])
    key_file.write_text(content)
    with pytest.raises(ValueError):
        indexnow.build_payload(sitemap, key_file)


class Response:
    def __init__(self, status, body=b""):
        self.status = status
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]


@pytest.mark.parametrize("key_status,key_body", [(404, KEY.encode()), (200, b"wrong"),
                                                (200, (KEY + "\n").encode())])
def test_failed_ownership_check_prevents_submission(indexnow, tmp_path, monkeypatch,
                                                   key_status, key_body):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL])
    payload = indexnow.build_payload(sitemap, key_file)
    requests = []

    def open_request(request, **kwargs):
        requests.append(request)
        return Response(key_status, key_body)

    monkeypatch.setattr(indexnow.OPENER, "open", open_request)
    with pytest.raises(ValueError):
        indexnow.submit(payload)
    assert len(requests) == 1
    assert requests[0].get_method() == "GET"
    assert requests[0].full_url == payload["keyLocation"]


@pytest.mark.parametrize("status", [200, 202])
def test_verified_key_allows_only_fixed_endpoint_notification(indexnow, tmp_path,
                                                            monkeypatch, status):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL])
    payload = indexnow.build_payload(sitemap, key_file)
    requests = []

    def open_request(request, **kwargs):
        requests.append(request)
        return Response(200, KEY.encode()) if len(requests) == 1 else Response(status)

    monkeypatch.setattr(indexnow.OPENER, "open", open_request)
    assert indexnow.submit(payload) == status
    assert len(requests) == 2
    assert requests[1].full_url == "https://api.indexnow.org/indexnow"
    assert requests[1].get_method() == "POST"
    assert json.loads(requests[1].data) == payload


def test_dry_run_never_opens_network(indexnow, tmp_path, monkeypatch, capsys):
    sitemap, key_file = write_inputs(tmp_path, [SITE_URL])

    def fail_on_network(*args, **kwargs):
        raise AssertionError("Dry runs must not make network requests")

    monkeypatch.setattr(indexnow.OPENER, "open", fail_on_network)
    assert indexnow.main(["--dry-run", "--sitemap", str(sitemap),
                          "--key-file", str(key_file)]) == 0
    assert json.loads(capsys.readouterr().out)["urlList"] == [SITE_URL]


def test_redirects_are_rejected(indexnow):
    handler = indexnow.NoRedirect()
    with pytest.raises(HTTPError):
        handler.redirect_request(None, None, 302, "Redirect", {}, "https://other.example/")
