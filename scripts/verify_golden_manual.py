#!/usr/bin/env python3
"""Verify the programmer-facing Superscalar NPU C++ manual."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


INTRINSIC_PAGE = re.compile(r"(?:t[a-z0-9_]+|mgather|mscatter)\.md")
INTRINSIC_BANNED = re.compile(
    r"\b(?:A2A3|A2/A3|A5|PTO-AS|IR Level|AS Level|SSA|DPS|RecordEvent|WaitEvents|TSYNC)\b",
    re.IGNORECASE,
)
PUBLIC_BANNED = re.compile(
    r"\b(?:Four-PE|PTO architecture|PTO intrinsic|two-level architecture|microbenchmark architecture)\b",
    re.IGNORECASE,
)


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "link", "script", "img"}:
            return
        wanted = "href" if tag in {"a", "link"} else "src"
        self.links.extend(value for name, value in attrs if name == wanted and value)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def catalog(root: Path) -> dict[str, object]:
    return json.loads(
        (root / "scripts/data/intrinsic-catalog.json").read_text(encoding="utf-8")
    )


def intrinsic_pages(docs: Path) -> dict[str, Path]:
    return {
        path.stem.upper(): path
        for path in (docs / "intrinsics").glob("*.md")
        if INTRINSIC_PAGE.fullmatch(path.name)
    }


def headings(text: str) -> list[str]:
    return [heading.strip().lower() for heading in re.findall(r"^##\s+(.+)$", text, re.MULTILINE)]


def check_intrinsics(root: Path, docs: Path, errors: list[str]) -> None:
    data = catalog(root)
    operations = data.get("operations", [])
    names = [str(entry["name"]) for entry in operations]
    if len(names) != 111 or len(set(names)) != 111:
        errors.append("intrinsic catalog must contain exactly 111 unique workbook operations")
    if data.get("workbook", {}).get("sha256") != "0387b39f108599f2469c2e2374a46ea4b832a96fee7bf7aa5d7cc575fdc7864a":
        errors.append("intrinsic catalog workbook digest differs from the publication authority")

    pages = intrinsic_pages(docs)
    if set(pages) != set(names):
        errors.append(
            "intrinsic page set differs from catalog: "
            f"missing={sorted(set(names) - set(pages))}, extra={sorted(set(pages) - set(names))}"
        )
    identities = {
        "get_thread_idx": docs / "intrinsics/get_thread_idx.md",
        "get_block_idx": docs / "intrinsics/get_block_idx.md",
    }
    for name, path in {**pages, **identities}.items():
        if not path.is_file():
            errors.append(f"missing public C++ entry page: {path.relative_to(root)}")
            continue
        text = path.read_text(encoding="utf-8")
        first = next((line for line in text.splitlines() if line.startswith("# ")), "")
        if not first.lower().startswith(f"# {name.lower()}"):
            errors.append(f"{path.relative_to(root)}: H1 must start with the API name")
        section_names = headings(text)
        required = {
            "purpose": lambda item: "purpose" in item,
            "declaration": lambda item: "declaration" in item,
            "parameters": lambda item: "parameter" in item,
            "semantics": lambda item: "semantic" in item or "operation" in item,
            "constraints": lambda item: "constraint" in item,
            "example": lambda item: "example" in item,
            "common mistakes": lambda item: "mistake" in item,
            "used by kernels": lambda item: "used by" in item,
        }
        for label, predicate in required.items():
            if not any(predicate(item) for item in section_names):
                errors.append(f"{path.relative_to(root)}: missing {label} section")
        if INTRINSIC_BANNED.search(text):
            errors.append(f"{path.relative_to(root)}: contains removed target or IR terminology")

    index = (docs / "intrinsics/index.md").read_text(encoding="utf-8")
    nav = (root / "mkdocs.yml").read_text(encoding="utf-8")
    for name in names:
        relative = f"intrinsics/{name.lower()}.md"
        if f"({name.lower()}.md)" not in index:
            errors.append(f"intrinsic index does not link {name}")
        if relative not in nav:
            errors.append(f"sidebar does not expose {name}")
    for name in identities:
        if f"({name}.md)" not in index or f"intrinsics/{name}.md" not in nav:
            errors.append(f"index or sidebar does not expose {name}")


def check_model(docs: Path, errors: list[str]) -> None:
    required = {
        "index.md": ("Superscalar NPU C++ Programming Manual",),
        "model/programming.md": ("block", "thread"),
        "model/core-indexing.md": ("get_thread_idx()", "get_block_idx()", "128"),
        "model/execution.md": (
            "tile version",
            "tile ID",
            "64 local tile registers",
            "TLOAD",
            "TSTORE",
            "shared tile-register file",
        ),
        "model/shared-tile-registers.md": ("TLOAD", "TSTORE", "TMOV", "TMATMUL"),
        "tutorials/group-programming.md": ("get_thread_idx()", "get_block_idx()"),
        "tutorials/multidimensional-tiling.md": ("LocalTile", "SharedTile", "tile row", "tile column"),
        "tutorials/fine-grained-tiles.md": ("128 bytes", "LocalTile", "TLOAD", "TSTORE"),
    }
    for relative, markers in required.items():
        path = docs / relative
        if not path.is_file():
            errors.append(f"missing manual page: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker.lower() not in text.lower():
                errors.append(f"{relative}: missing programming-model marker {marker!r}")

    all_model_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for base in (docs / "model", docs / "cpp")
        for path in base.glob("*.md")
    )
    if re.search(r"\b(?:RecordEvent|WaitEvents|TSYNC)\b", all_model_text, re.IGNORECASE):
        errors.append("programming model retains a removed completion API")


def check_benchmarks(root: Path, docs: Path, errors: list[str]) -> None:
    generator = load_module(root / "scripts/generate_benchmark_manual.py", "benchmark_manual")
    builds = generator.inventory(root)
    path = docs / "benchmarks/catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("builds", [])
    if len(rows) != len(builds):
        errors.append(f"benchmark catalog has {len(rows)} builds; manifests have {len(builds)}")
    actual = {(generator.rel(item.manifest, root), item.line, item.command) for item in builds}
    generated = {(row["manifest"], row["manifest_line"], row["command"]) for row in rows}
    if actual != generated:
        errors.append("benchmark catalog commands differ from active manifests")

    pages = {str(row["page"]) for row in rows}
    for relative in pages:
        page = docs / relative
        if not page.is_file():
            errors.append(f"missing generated benchmark page: {relative}")
            continue
        text = page.read_text(encoding="utf-8")
        for heading in ("Kernel structure", "Core kernel", "Intrinsic sequence", "Active Build Commands"):
            if f"## {heading}" not in text:
                errors.append(f"{relative}: missing {heading} section")


def prose(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\[[^]]+\]\([^)]*\)", "", text)
    return text


def check_public_surface(root: Path, docs: Path, errors: list[str]) -> None:
    nav = (root / "mkdocs.yml").read_text(encoding="utf-8")
    if PUBLIC_BANNED.search(nav):
        errors.append("sidebar exposes removed architecture branding")
    for path in [docs / "index.md", *docs.joinpath("model").glob("*.md"), *docs.joinpath("cpp").glob("*.md")]:
        if PUBLIC_BANNED.search(prose(path.read_text(encoding="utf-8", errors="replace"))):
            errors.append(f"{path.relative_to(root)}: exposes removed architecture branding")
    developer_home = Path.home().as_posix().rstrip("/") + "/"
    for path in [root / "README.md", root / "mkdocs.yml", *docs.rglob("*.md")]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if developer_home in text:
            errors.append(f"{path.relative_to(root)}: contains a developer-local path")

    reader_files = [*docs.rglob("*.md"), *docs.joinpath("examples").glob("*.cpp")]
    for path in reader_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "pto::linx" in text or "pto/linx" in text:
            errors.append(
                f"{path.relative_to(root)}: exposes a backend namespace in reader-facing code"
            )

    learning_files = [
        *docs.joinpath("start").glob("*.md"),
        *docs.joinpath("cpp").glob("*.md"),
        *docs.joinpath("model").glob("*.md"),
        *docs.joinpath("tutorials").glob("*.md"),
    ]
    for path in learning_files:
        if "Location::Vec" in path.read_text(encoding="utf-8", errors="replace"):
            errors.append(
                f"{path.relative_to(root)}: uses an implementation tile spelling instead of LocalTile"
            )

    fine_example = docs / "examples/fine_grained_tiles.cpp"
    if not fine_example.is_file():
        errors.append("missing compile-checked fine-grained tile example")
    else:
        text = fine_example.read_text(encoding="utf-8")
        for marker in ("kFineTileBytes = 128", "F32VectorTile", "F32MatrixTile", "TROWSUM"):
            if marker not in text:
                errors.append(f"fine-grained tile example is missing {marker!r}")


def resolve_site_target(site: Path, page: Path, link: str) -> Path | None:
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc or link.startswith(("#", "mailto:", "javascript:")):
        return None
    raw = unquote(parsed.path)
    if not raw:
        return None
    site_prefix = "/SuperNPUBench"
    if raw == site_prefix or raw.startswith(site_prefix + "/"):
        raw = raw[len(site_prefix):] or "/"
    if raw in {"/.", "/./"}:
        raw = "/"
    target = site / raw.lstrip("/") if raw.startswith("/") else page.parent / raw
    target = target.resolve()
    if raw.endswith("/"):
        return target / "index.html"
    if target.suffix:
        return target
    return target if target.exists() else target / "index.html"


def check_site(site: Path, errors: list[str]) -> None:
    if not site.is_dir():
        errors.append(f"built site directory missing: {site}")
        return
    for html in site.rglob("*.html"):
        parser = Links()
        parser.feed(html.read_text(encoding="utf-8", errors="replace"))
        for link in parser.links:
            target = resolve_site_target(site, html, link)
            if target is not None and not target.exists():
                errors.append(f"{html.relative_to(site)}: broken link {link}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    docs = root / "docs"
    errors: list[str] = []
    check_intrinsics(root, docs, errors)
    check_model(docs, errors)
    check_benchmarks(root, docs, errors)
    check_public_surface(root, docs, errors)
    from verify_pto_kernel_migration import check_migration as check_foundation_migration
    from verify_deepseek_migration import check_migration as check_deepseek_migration

    check_foundation_migration(root, errors)
    check_deepseek_migration(root, errors)
    check_site((args.site or root / "site").resolve(), errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"manual verification failed with {len(errors)} errors")
    print(
        "Manual verified: 113 public C++ entries, 37 pinned DeepSeek modules, "
        f"{len(json.loads((docs / 'benchmarks/catalog.json').read_text())['builds'])} benchmark builds, links clean."
    )


if __name__ == "__main__":
    main()
