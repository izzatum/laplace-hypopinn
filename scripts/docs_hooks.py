"""Build-time metadata and repository-link adaptation for the static docs site.

Scientific Markdown remains readable directly on GitHub. Only links to files
outside the published site are rewritten; no source or archive is changed.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path
import posixpath
import re
import struct
import tomllib
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlsplit, urlunsplit

from mkdocs.structure.files import File

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Inline links/images and reference definitions in these hand-authored documents.
_INLINE_LINK = re.compile(r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<url>[^\s)]+)")
_REFERENCE_LINK = re.compile(r"(?m)^(?P<prefix>\s*\[[^\]\n]+\]:\s*)(?P<url>\S+)")
_FENCED_CODE = re.compile(r"(^\s*```.*?^\s*```\s*$|^\s*~~~.*?^\s*~~~\s*$)", re.M | re.S)


def on_config(config):
    """Read public verification tokens without inserting empty ownership claims."""
    config.extra["google_site_verification"] = os.environ.get(
        "GOOGLE_SITE_VERIFICATION", ""
    ).strip()
    config.extra["bing_site_verification"] = os.environ.get("BING_SITE_VERIFICATION", "").strip()
    return config


def _repository_link(url: str, source_uri: str, repo_url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
        return url
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_uri), parsed.path))
    if resolved == ".." or resolved.startswith("../"):
        # Paths are resolved relative to docs/, as they are in the repository.
        repo_path = posixpath.normpath(posixpath.join("docs", resolved))
    elif resolved == "superpowers" or resolved.startswith("superpowers/"):
        repo_path = "docs/" + resolved
    else:
        return url
    target = _REPOSITORY_ROOT / repo_path
    if not target.exists():
        raise ValueError(f"Documentation link target does not exist: {source_uri}: {url}")
    kind = "tree" if target.is_dir() else "blob"
    destination = f"{repo_url.rstrip('/')}/{kind}/main/{quote(repo_path, safe='/')}"
    return urlunsplit((*urlsplit(destination)[:3], parsed.query, parsed.fragment))


def on_page_markdown(markdown, page, config, files):
    """Keep source-relative links useful in both GitHub and the public site."""
    def replace(match):
        return match["prefix"] + _repository_link(
            match["url"], page.file.src_uri, config.repo_url
        )

    # Preserve links shown as example code instead of silently rewriting examples.
    sections = _FENCED_CODE.split(markdown)
    for index in range(0, len(sections), 2):
        sections[index] = _REFERENCE_LINK.sub(
            replace, _INLINE_LINK.sub(replace, sections[index])
        )
    return "".join(sections)


def on_page_context(context, page, config, nav):
    """Describe only factual software and navigation entities, without ratings."""
    canonical = page.canonical_url
    if not canonical:
        context["discovery_structured_data"] = []
        return context
    if page.is_homepage:
        project = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text())["project"]
        context["discovery_structured_data"] = [{
            "@context": "https://schema.org",
            "@type": "SoftwareSourceCode",
            "@id": canonical + "#software",
            "name": "Laplace HypoPINN",
            "description": page.meta.get("description", config.site_description),
            "url": canonical,
            "codeRepository": config.repo_url,
            "programmingLanguage": "Python",
            "runtimePlatform": "Python >=3.12; PyTorch",
            "softwareVersion": project["version"],
            "license": config.repo_url + "/blob/main/LICENSE",
            "usageInfo": config.repo_url + "/blob/main/THIRD_PARTY_NOTICES.md",
            "citation": {
                "@type": "ScholarlyArticle",
                "name": (
                    "Laplace HypoPINN: physics-informed neural network for hypocenter "
                    "localization and its predictive uncertainty"
                ),
                "url": "https://doi.org/10.1088/2632-2153/ac94b3",
                "identifier": "https://doi.org/10.1088/2632-2153/ac94b3",
            },
        }]
    else:
        items = [{"@type": "ListItem", "position": 1,
                  "name": config.site_name, "item": config.site_url}]
        if page.file.src_uri.startswith("tutorials/"):
            items.append({"@type": "ListItem", "position": 2,
                          "name": "Tutorials", "item": config.site_url + "tutorials/"})
        items.append({"@type": "ListItem", "position": len(items) + 1,
                      "name": page.title, "item": canonical})
        context["discovery_structured_data"] = [{
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": items,
        }]
    return context


_NOTEBOOK_METHODS = (
    ("01_legacy_diagonal", "Historical diagonal", "batch-gradient-square curvature"),
    ("02_diagonal", "GGN diagonal", "diagonal generalized Gauss–Newton curvature"),
    ("03_block_diagonal", "Block diagonal", "layer-block generalized Gauss–Newton curvature"),
    ("04_full", "Full GGN", "full generalized Gauss–Newton curvature"),
    ("05_kfac", "KFAC", "Kronecker-factored curvature"),
    ("06_efb", "EFB", "an eigenvalue-corrected Kronecker basis"),
    ("07_inf", "INF", "sparse information-form curvature"),
    ("08_valla", "VaLLA", "a variational linearized Laplace approximation"),
)
_FIGURE_DESCRIPTIONS = (
    "Known velocity field, receiver geometry and reference source location",
    "Training objectives and observed versus predicted receiver travel times",
    "Analytic reference travel times, trained PINN field and signed field error",
    "Method-specific curvature and posterior approximation diagnostics",
    "Sampled travel-time standard deviation and source-location samples",
    "Travel-time cross-section with reference, trained center and descriptive sample spread",
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _joined(value):
    return "".join(value) if isinstance(value, list) else value


def _code_block(source, language="text"):
    fence = "`" * max(3, max((len(x) + 1 for x in re.findall(r"`+", source)), default=3))
    return f"\n{fence}{language}\n{source.rstrip()}\n{fence}\n"


def _notebook_markdown(markdown, notebook, output_uri, repo_url):
    """Translate notebook-relative links to the same research sources on the site."""
    def replace(match):
        url = match["url"]
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
            return match[0]
        resolved = posixpath.normpath(posixpath.join(notebook.parent.as_posix(), parsed.path))
        if resolved == "examples/notebooks/README.md":
            destination = posixpath.relpath("tutorials.md", posixpath.dirname(output_uri))
        elif resolved.startswith("docs/") and not resolved.startswith("docs/superpowers/"):
            destination = posixpath.relpath(resolved[5:], posixpath.dirname(output_uri))
        else:
            target = _REPOSITORY_ROOT / resolved
            if not target.exists():
                raise ValueError(f"Broken notebook link in {notebook}: {url}")
            kind = "tree" if target.is_dir() else "blob"
            destination = f"{repo_url}/{kind}/main/{quote(resolved, safe='/')}"
        rewritten = urlunsplit((*urlsplit(destination)[:3], parsed.query, parsed.fragment))
        return match["prefix"] + rewritten

    sections = _FENCED_CODE.split(markdown)
    for index in range(0, len(sections), 2):
        sections[index] = _REFERENCE_LINK.sub(
            replace, _INLINE_LINK.sub(replace, sections[index])
        )
    return "".join(sections)


def on_files(files, config):
    """Publish existing executed tutorial cells as static, readable lesson pages.

    This intentionally does not execute notebooks or render arbitrary HTML output.
    Each figure is copied from the committed PNG output; text and code remain
    ordinary HTML after Markdown conversion, readable without running JavaScript.
    """
    for stem, label, convention in _NOTEBOOK_METHODS:
        notebook_path = Path("examples/notebooks") / f"{stem}.ipynb"
        notebook = json.loads((_REPOSITORY_ROOT / notebook_path).read_text())
        method = stem.split("_", 1)[1]
        output_uri = f"tutorials/{method}.md"
        description = (
            f"Run a {label} Laplace tutorial for a physics-informed neural network: "
            f"fit {convention}, sample fields and inspect saved Python outputs and plots."
        )
        content = [
            "---\n"
            f"title: {json.dumps(label + ' Laplace approximation tutorial')}\n"
            f"description: {json.dumps(description)}\n"
            "---\n",
        ]
        figure_index = 0
        for index, cell in enumerate(notebook["cells"]):
            source = _joined(cell["source"])
            if cell["cell_type"] == "markdown":
                content.append(_notebook_markdown(source, notebook_path, output_uri, config.repo_url))
                if index == 0:
                    content.append(
                        "!!! note \"Read here; run the notebook locally\"\n\n"
                        "    This page displays the committed notebook's code, saved output and "
                        "six figures. Documentation builds do not rerun experiments. "
                        "The recorded results use a short teaching budget.\n\n"
                        f"    [View the source notebook]({config.repo_url}/blob/main/{notebook_path}) "
                        f"or [download the notebook](https://raw.githubusercontent.com/"
                        f"izzatum/laplace-hypopinn/main/{notebook_path}) to run it.\n"
                    )
            elif cell["cell_type"] == "code":
                content.append(_code_block(source, "python"))
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        raise ValueError(f"Saved notebook contains an error: {notebook_path}")
                    data = output.get("data", {})
                    if "image/png" in data:
                        png = base64.b64decode(_joined(data["image/png"]), validate=True)
                        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise ValueError(f"Invalid PNG in {notebook_path}")
                        image_uri = f"assets/tutorials/{method}-{figure_index + 1:02}.png"
                        files.append(File.generated(config, image_uri, content=png))
                        if figure_index >= len(_FIGURE_DESCRIPTIONS):
                            raise ValueError(f"More than six saved figures in {notebook_path}")
                        caption = f"{label}: {_FIGURE_DESCRIPTIONS[figure_index]}"
                        width, height = struct.unpack(">II", png[16:24])
                        content.append(
                            f"![{caption}](../{image_uri})"
                            f"{{ loading=lazy width={width} height={height} }}\n"
                        )
                        figure_index += 1
                    elif "text/markdown" in data:
                        content.append(_notebook_markdown(
                            _joined(data["text/markdown"]), notebook_path, output_uri, config.repo_url
                        ))
                    elif "text/plain" in data:
                        content.append(_code_block(_ANSI_ESCAPE.sub("", _joined(data["text/plain"]))))
                    elif output.get("output_type") == "stream":
                        content.append(_code_block(_ANSI_ESCAPE.sub("", _joined(output["text"]))))
            else:
                raise ValueError(f"Unsupported tutorial cell: {notebook_path}: {cell['cell_type']}")
        if figure_index != len(_FIGURE_DESCRIPTIONS):
            raise ValueError(f"Expected six saved figures in {notebook_path}; got {figure_index}")
        lesson = File.generated(config, output_uri, content="\n\n".join(content))
        lesson.edit_uri = "../" + notebook_path.as_posix()
        files.append(lesson)
    return files


def on_post_build(config):
    """Do not label unchanged content as new on every documentation rebuild.

    MkDocs' default sitemap lastmod is the build date. Omit it until genuine
    per-page modification dates are available; keep the gzip sitemap identical.
    """
    path = Path(config.site_dir) / "sitemap.xml"
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", namespace)
    tree = ET.parse(path)
    for entry in tree.getroot():
        for lastmod in list(entry.findall(f"{{{namespace}}}lastmod")):
            entry.remove(lastmod)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    path.with_suffix(".xml.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
