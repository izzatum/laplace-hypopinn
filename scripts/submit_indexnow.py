#!/usr/bin/env python3
"""Notify IndexNow about this project's built sitemap after verifying its public key.

Use ``--dry-run`` to print the payload without any network requests. Acceptance is
only a change notification; it does not guarantee indexing or search rankings.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SITE_URL = "https://izzatum.github.io/laplace-hypopinn/"
KEY_FILE = Path(__file__).resolve().parents[1] / "docs/00e2a02d567caa5df2b7c97072b4eb95.txt"
ENDPOINT = "https://api.indexnow.org/indexnow"
NAMESPACE = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class NoRedirect(HTTPRedirectHandler):
    """Keep ownership checks and submissions at their explicit URLs."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(newurl, code, "IndexNow redirects are disabled", headers, fp)


OPENER = build_opener(NoRedirect())


def canonical_url(value: str):
    """Accept unambiguous HTTPS URLs without credentials or alternate encodings."""
    if not value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"Invalid URL: {value!r}")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or "\\" in value or "%" in value
            or any(part in (".", "..") for part in parsed.path.split("/"))
            or "//" in parsed.path):
        raise ValueError(f"Use a canonical HTTPS URL without query or fragment: {value!r}")
    # Accessing port also rejects malformed or out-of-range port numbers.
    parsed.port
    return parsed


def build_payload(sitemap: Path, key_file: Path, site_url: str = SITE_URL) -> dict:
    """Read a URL sitemap and constrain every URL to the configured project path."""
    site = canonical_url(site_url)
    if not site.path.startswith("/") or site.path == "/" or not site.path.endswith("/"):
        raise ValueError("The site URL must have a non-root project path ending in '/'")
    key = key_file.read_text(encoding="ascii")
    if not re.fullmatch(r"[A-Za-z0-9-]{8,128}", key) or key_file.name != f"{key}.txt":
        raise ValueError("Key file must contain only its 8–128 character key and be named <key>.txt")
    root = ET.parse(sitemap).getroot()
    if root.tag != NAMESPACE + "urlset":
        raise ValueError("Expected a sitemap URL set, not a sitemap index")
    urls = [entry.findtext(NAMESPACE + "loc") for entry in root.findall(NAMESPACE + "url")]
    if not 1 <= len(urls) <= 10_000 or len(set(urls)) != len(urls):
        raise ValueError("Sitemap must contain 1–10,000 unique URLs")
    for url in urls:
        parsed = canonical_url(url)
        if (parsed.scheme, parsed.netloc) != (site.scheme, site.netloc) \
                or not parsed.path.startswith(site.path):
            raise ValueError(f"Sitemap URL is outside the project scope: {url!r}")
    return {
        "host": site.netloc,
        "key": key,
        "keyLocation": site_url + key_file.name,
        "urlList": urls,
    }


def submit(payload: dict) -> int:
    """Verify the live key, then submit one notification without following redirects."""
    expected = payload["key"].encode("ascii")
    with OPENER.open(Request(payload["keyLocation"]), timeout=30) as response:
        if response.status != 200 or response.read(len(expected) + 1) != expected:
            raise ValueError("Published ownership key must return HTTP 200 with exact key content")
    request = Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with OPENER.open(request, timeout=30) as response:
        if response.status not in (200, 202):
            raise ValueError(f"IndexNow did not accept the notification (HTTP {response.status})")
        return response.status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sitemap", type=Path, default=Path("site/sitemap.xml"))
    parser.add_argument("--site-url", default=SITE_URL)
    parser.add_argument("--key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Print JSON without network requests")
    args = parser.parse_args(argv)
    try:
        payload = build_payload(args.sitemap, args.key_file, args.site_url)
        if args.dry_run:
            print(json.dumps(payload, indent=2))
        else:
            status = submit(payload)
            suffix = " (ownership validation pending)" if status == 202 else ""
            print(f"IndexNow accepted {len(payload['urlList'])} URL notifications: HTTP {status}{suffix}.")
            print("Notification acceptance does not guarantee indexing or search rankings.")
    except (OSError, ValueError, ET.ParseError, URLError) as error:
        print(f"IndexNow: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
