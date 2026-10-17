#!/usr/bin/env python3
"""Check the built documentation's crawl metadata, structured data, and local links.

Run ``python scripts/check_docs.py [site]`` after ``mkdocs build --strict``.
This checker deliberately uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


SITE_URL = "https://izzatum.github.io/laplace-hypopinn/"
REPOSITORY_URL = "https://github.com/izzatum/laplace-hypopinn"
TUTORIAL_METHODS = (
    "legacy_diagonal", "diagonal", "block_diagonal", "full", "kfac", "efb", "inf", "valla",
)
MIT_URLS = {
    "https://opensource.org/license/mit",
    "https://opensource.org/licenses/MIT",
    "https://spdx.org/licenses/MIT.html",
    REPOSITORY_URL + "/blob/main/LICENSE",
}


@dataclass
class Page:
    """Only the generated HTML facts needed for site validation."""

    title: str = ""
    metadata: dict[str, list[str]] = field(default_factory=dict)
    canonicals: list[str] = field(default_factory=list)
    anchors: set[str] = field(default_factory=set)
    references: list[str] = field(default_factory=list)
    structured_data: list[str] = field(default_factory=list)
    images: list[tuple[str, str]] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = Page()
        self.in_title = False
        self.in_json = False
        self.json_parts: list[str] = []
        self.in_pre = False
        self.in_code = False
        self.code_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.page.anchors.add(values["id"])
        if tag == "a" and values.get("name"):
            self.page.anchors.add(values["name"])
        for attribute in ("href", "src", "poster"):
            if values.get(attribute):
                self.page.references.append(values[attribute])
        if tag == "img":
            self.page.images.append((values.get("src") or "", values.get("alt") or ""))
        if tag == "pre":
            self.in_pre = True
        elif tag == "code" and self.in_pre:
            self.in_code = True
            self.code_parts = []
        if tag == "title":
            self.in_title = True
        elif tag == "meta":
            key = values.get("name") or values.get("property")
            if key:
                self.page.metadata.setdefault(key.lower(), []).append(values.get("content") or "")
        elif tag == "link" and "canonical" in (values.get("rel") or "").lower().split():
            self.page.canonicals.append(values.get("href") or "")
        elif tag == "script" and values.get("type") == "application/ld+json":
            self.in_json = True
            self.json_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag == "script" and self.in_json:
            self.page.structured_data.append("".join(self.json_parts))
            self.in_json = False
        elif tag == "code" and self.in_code:
            self.page.code_blocks.append("".join(self.code_parts))
            self.in_code = False
        elif tag == "pre":
            self.in_pre = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.page.title += data
        if self.in_json:
            self.json_parts.append(data)
        if self.in_code:
            self.code_parts.append(data)


def page_url(relative_path: Path) -> str:
    """Map MkDocs' index pages to their published directory URLs."""
    relative = relative_path.as_posix()
    if relative_path.name == "index.html":
        relative = relative[: -len("index.html")]
    return urljoin(SITE_URL, relative)


def schema_nodes(value: object) -> list[dict]:
    """Accept either individual schema objects or JSON-LD graphs."""
    if isinstance(value, list):
        return [node for item in value for node in schema_nodes(item)]
    if isinstance(value, dict):
        graph = value.get("@graph")
        return [value, *schema_nodes(graph)] if graph is not None else [value]
    return []


class SiteChecker:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.errors: list[str] = []
        self.pages: dict[Path, Page] = {}

    def fail(self, location: Path | str, message: str) -> None:
        self.errors.append(f"{location}: {message}")

    def check_reference(self, source: Path, reference: str) -> bool:
        """Validate a reference; return whether it points into this website."""
        try:
            resolved = urlsplit(urljoin(page_url(source), reference))
        except ValueError as exc:
            self.fail(source, f"malformed URL {reference!r}: {exc}")
            return False
        site = urlsplit(SITE_URL)
        if resolved.scheme not in {"http", "https"} or resolved.netloc != site.netloc:
            return False
        if not resolved.path.startswith(site.path):
            self.fail(source, f"local URL escapes the project site path: {reference!r}")
            return True
        relative = unquote(resolved.path[len(site.path) :])
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            self.fail(source, f"local URL escapes the build directory: {reference!r}")
            return True
        if target.is_dir():
            target /= "index.html"
        if not target.is_file():
            self.fail(source, f"missing local target: {reference!r}")
            return True
        if resolved.fragment and target.suffix == ".html":
            page = self.pages.get(target.relative_to(self.root))
            if page is not None and unquote(resolved.fragment) not in page.anchors:
                self.fail(source, f"missing local anchor: {reference!r}")
        return True

    def check_schema(self, relative: Path, page: Page) -> None:
        nodes = []
        for raw in page.structured_data:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                self.fail(relative, f"invalid JSON-LD: {exc}")
                continue
            parsed = schema_nodes(value)
            if not parsed or (
                isinstance(value, list) and not all(isinstance(item, dict) for item in value)
            ):
                self.fail(relative, "JSON-LD must contain schema objects or an @graph")
                continue
            for node in parsed:
                if "@type" not in node and "@graph" not in node:
                    self.fail(relative, "JSON-LD object is missing @type or @graph")
                if "@graph" in node and not isinstance(node["@graph"], (list, dict)):
                    self.fail(relative, "JSON-LD @graph must contain schema objects")
                if "@type" in node:
                    types = node["@type"] if isinstance(node["@type"], list) else [node["@type"]]
                    if not types or not all(isinstance(item, str) and item for item in types):
                        self.fail(relative, "JSON-LD @type must be a nonempty string or string list")
            nodes.extend(parsed)
        if relative != Path("index.html"):
            return
        software = [
            node for node in nodes
            if "SoftwareSourceCode" in (
                node.get("@type") if isinstance(node.get("@type"), list)
                else [node.get("@type")]
            )
        ]
        if len(software) != 1:
            self.fail(relative, "expected exactly one SoftwareSourceCode JSON-LD object")
            return
        node = software[0]
        for key, expected in {
            "codeRepository": REPOSITORY_URL,
            "programmingLanguage": "Python",
            "softwareVersion": "0.1.0",
        }.items():
            if node.get(key) != expected:
                self.fail(relative, f"SoftwareSourceCode {key} must be {expected!r}")
        license_url = node.get("license")
        if not isinstance(license_url, str) or license_url.rstrip("/") not in MIT_URLS:
            self.fail(relative, "SoftwareSourceCode license must be a canonical MIT license URL")
        elif license_url == REPOSITORY_URL + "/blob/main/LICENSE":
            license_path = Path(__file__).resolve().parents[1] / "LICENSE"
            try:
                license_text = license_path.read_text(encoding="utf-8")
            except OSError as exc:
                self.fail(relative, f"cannot verify the repository's MIT license: {exc}")
            else:
                if not all(phrase in license_text for phrase in (
                    "MIT License", "Permission is hereby granted, free of charge",
                )):
                    self.fail(relative, "repository LICENSE does not contain the MIT license grant")

    def check_pages(self) -> set[str]:
        titles: dict[str, Path] = {}
        descriptions: dict[str, Path] = {}
        canonicals = set()
        for relative, page in self.pages.items():
            if relative.name == "404.html" or relative.parts[0] in {"assets", "search"}:
                continue
            expected_url = page_url(relative)
            if page.canonicals != [expected_url]:
                self.fail(relative, f"expected one canonical URL equal to {expected_url!r}")
            canonicals.add(expected_url)
            for label, value, seen in (
                ("title", page.title, titles),
                ("description", " ".join(page.metadata.get("description", [])), descriptions),
            ):
                normalized = " ".join(value.split()).casefold()
                if not normalized:
                    self.fail(relative, f"missing or empty {label}")
                elif normalized in seen:
                    self.fail(relative, f"duplicate {label}, also used by {seen[normalized]}")
                else:
                    seen[normalized] = relative
            if len(page.metadata.get("description", [])) != 1:
                self.fail(relative, "expected exactly one meta description")
            for key in (
                "og:title", "og:description", "og:type", "og:url",
                "twitter:card", "twitter:title", "twitter:description",
            ):
                if len(page.metadata.get(key, [])) != 1 or not page.metadata[key][0].strip():
                    self.fail(relative, f"expected one nonempty {key} meta tag")
            if page.metadata.get("og:url") != [expected_url]:
                self.fail(relative, "og:url must equal the canonical URL")
            for key in ("robots", "googlebot", "bingbot"):
                directives = re.split(r"[\s,]+", " ".join(page.metadata.get(key, [])).lower())
                if "noindex" in directives or "none" in directives:
                    self.fail(relative, f"{key} blocks indexing of a public page")
            for reference in set(page.references):
                self.check_reference(relative, reference)
            for key in ("og:image", "twitter:image"):
                for reference in page.metadata.get(key, []):
                    self.check_reference(relative, reference)
            self.check_schema(relative, page)
        return canonicals

    def check_sitemap(self, canonicals: set[str]) -> None:
        path = self.root / "sitemap.xml"
        try:
            tree = ET.parse(path)
        except (OSError, ET.ParseError) as exc:
            self.fail("sitemap.xml", str(exc))
            return
        urls = [node.text or "" for node in tree.iter() if node.tag.rsplit("}", 1)[-1] == "loc"]
        if any(node.tag.rsplit("}", 1)[-1] == "lastmod" for node in tree.iter()):
            self.fail("sitemap.xml", "omit lastmod until it reflects actual content modification dates")
        if len(urls) != len(set(urls)):
            self.fail("sitemap.xml", "duplicate URLs")
        for missing in sorted(canonicals - set(urls)):
            self.fail("sitemap.xml", f"missing public page: {missing}")
        for extra in sorted(set(urls) - canonicals):
            self.fail("sitemap.xml", f"URL does not match a public page canonical: {extra}")
        try:
            compressed = gzip.decompress((self.root / "sitemap.xml.gz").read_bytes())
        except (OSError, EOFError, zlib.error) as exc:
            self.fail("sitemap.xml.gz", f"cannot read the compressed sitemap: {exc}")
        else:
            if compressed != path.read_bytes():
                self.fail("sitemap.xml.gz", "decompressed content must exactly match sitemap.xml")

    def check_tutorials(self) -> None:
        for method in TUTORIAL_METHODS:
            relative = Path("tutorials") / method / "index.html"
            page = self.pages.get(relative)
            if page is None:
                self.fail(relative, "missing method tutorial")
                continue
            images = []
            for source, alt in page.images:
                try:
                    resolved = urlsplit(urljoin(page_url(relative), source))
                except ValueError:
                    continue  # Reported by the generic link check.
                if resolved.path.lower().endswith(".png"):
                    images.append((resolved, alt))
            if len(images) != 6 or len({url.geturl() for url, _ in images}) != 6:
                self.fail(relative, "expected six distinct saved PNG figures")
            alternatives = set()
            for url, alt in images:
                normalized = " ".join(alt.split()).casefold()
                if len(re.findall(r"[\w-]+", normalized)) < 5 or len(normalized) < 25:
                    self.fail(relative, f"figure needs descriptive alt text: {url.path}")
                if normalized in alternatives:
                    self.fail(relative, f"figures need distinct alt text: {alt!r}")
                alternatives.add(normalized)
                site = urlsplit(SITE_URL)
                if url.netloc != site.netloc or not url.path.startswith(site.path):
                    self.fail(relative, f"saved tutorial figure must be local: {url.geturl()}")
                    continue
                target = (self.root / unquote(url.path[len(site.path) :])).resolve()
                if not target.is_relative_to(self.root):
                    continue  # Reported by the generic link check.
                try:
                    with target.open("rb") as handle:
                        header = handle.read(24)
                        trailer = b""
                        if len(header) == 24:
                            handle.seek(-12, 2)
                            trailer = handle.read(12)
                except OSError as exc:
                    self.fail(relative, f"cannot read saved PNG {target.name}: {exc}")
                    continue
                if (
                    len(header) != 24
                    or not header.startswith(b"\x89PNG\r\n\x1a\n")
                    or header[12:16] != b"IHDR"
                    or not all(struct.unpack(">II", header[16:24]))
                    or trailer != b"\x00\x00\x00\x00IEND\xaeB`\x82"
                ):
                    self.fail(relative, f"figure must be a nonempty PNG: {target.name}")
            has_python = False
            for block in page.code_blocks:
                source = "\n".join(line for line in block.splitlines() if not line.startswith("%"))
                try:
                    module = ast.parse(source)
                except SyntaxError:
                    continue  # Saved output blocks need not be Python.
                for node in ast.walk(module):
                    if isinstance(node, ast.ImportFrom):
                        has_python |= (node.module or "").split(".")[0] == "laplace_hypopinn"
                    elif isinstance(node, ast.Import):
                        has_python |= any(
                            alias.name.split(".")[0] == "laplace_hypopinn" for alias in node.names
                        )
            if not has_python:
                self.fail(relative, "expected rendered Python code importing laplace_hypopinn")

    def check_llms(self) -> None:
        path = self.root / "llms.txt"
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError as exc:
            self.fail("llms.txt", str(exc))
            return
        references = set(re.findall(r"\[[^\]]*\]\(\s*<?([^\s)>]+)", contents))
        references.update(url.rstrip(".,;") for url in re.findall(r"https?://[^\s<>)]+", contents))
        local_count = sum(self.check_reference(Path("llms.txt"), url) for url in references)
        if not local_count:
            self.fail("llms.txt", "expected at least one link to local documentation")

    def run(self) -> int:
        if not self.root.is_dir():
            self.fail(self.root, "build directory not found; run mkdocs build --strict first")
        for path in sorted(self.root.rglob("*.html")):
            parser = PageParser()
            try:
                parser.feed(path.read_text(encoding="utf-8"))
                parser.close()
            except (OSError, UnicodeError) as exc:
                self.fail(path.relative_to(self.root), f"cannot parse HTML: {exc}")
                continue
            self.pages[path.relative_to(self.root)] = parser.page
        if Path("index.html") not in self.pages:
            self.fail("index.html", "missing homepage")
        canonicals = self.check_pages()
        self.check_sitemap(canonicals)
        self.check_llms()
        self.check_tutorials()
        if self.errors:
            print(f"Documentation checks failed ({len(self.errors)}):", file=sys.stderr)
            for error in self.errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print(
            f"Documentation checks passed: {len(canonicals)} pages, eight tutorials, "
            "sitemaps, JSON-LD, and llms.txt."
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", type=Path, default=Path("site"))
    return SiteChecker(parser.parse_args().site).run()


if __name__ == "__main__":
    raise SystemExit(main())
