#!/usr/bin/env python3
"""Verify the LinxISA PTO 0.57 one-level golden manual."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


INSTRUCTION_FILE = re.compile(r"(?:t[a-z0-9_]+|mgather|mscatter)\.md")
REMOVED_COMPLETION_API = re.compile(
    r"\b(?:event|events|RecordEvent|WaitEvents|TSYNC)\b", re.IGNORECASE
)
REMOVED_WEBSITE_SURFACE = re.compile(
    r"\b(?:two-level|microbenchmark)\b|two-level-arch|intrinsics/communication",
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
        for name, value in attrs:
            if name == wanted and value:
                self.links.append(value)


def load_generator(root: Path):
    path = root / "scripts" / "generate_benchmark_manual.py"
    spec = importlib.util.spec_from_file_location("benchmark_manual", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_allowlist(root: Path) -> list[str]:
    path = root / "scripts" / "data" / "linxisa-0.57-intrinsics.txt"
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(names) != 111 or len(set(names)) != 111:
        raise ValueError(f"{path}: expected 111 unique intrinsic names")
    return names


def instruction_pages(docs: Path) -> dict[str, Path]:
    return {
        path.stem.upper(): path
        for path in (docs / "intrinsics").glob("*.md")
        if INSTRUCTION_FILE.fullmatch(path.name)
    }


def check_intrinsics(root: Path, docs: Path, errors: list[str]) -> None:
    allowed = set(read_allowlist(root))
    pages = instruction_pages(docs)
    actual = set(pages)
    if actual != allowed:
        errors.append(
            "intrinsic page set differs from LinxISA PTO 0.57 allowlist: "
            f"missing={sorted(allowed - actual)}, extra={sorted(actual - allowed)}"
        )
    if (docs / "intrinsics" / "communication").exists():
        errors.append("removed communication intrinsic subtree still exists")

    identities = {
        "get_thread_idx": docs / "intrinsics" / "get_thread_idx.md",
        "get_block_idx": docs / "intrinsics" / "get_block_idx.md",
    }
    for name, path in {**pages, **identities}.items():
        if not path.is_file():
            errors.append(f"missing public intrinsic page: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for heading in ("Introduction", "C++ Intrinsic", "Constraints"):
            if not re.search(rf"^## {re.escape(heading)}\s*$", text, re.MULTILINE):
                errors.append(f"{path}: missing {heading} section")
        if not re.search(r"^## Examples?\s*$", text, re.MULTILINE):
            errors.append(f"{path}: missing example section")
        if re.search(r"\bTODO\b|task_progress|write_to_file", text, re.IGNORECASE):
            errors.append(f"{path}: contains unfinished authoring content")
        if re.search(r"\ufffd|Ã|Â|â[€™œ“”‘’]|ðŸ", text):
            errors.append(f"{path}: contains mojibake")
        if re.search(r"docs/(?:isa|coding)/", text):
            errors.append(f"{path}: contains a stale manual-internal path")

    thread_text = identities["get_thread_idx"].read_text(encoding="utf-8")
    block_text = identities["get_block_idx"].read_text(encoding="utf-8")
    for marker in ("same program entry and initial PC", "[0, 3]", "own architectural PC"):
        if marker not in thread_text:
            errors.append(f"get_thread_idx contract lacks {marker!r}")
    for marker in ("core ID", "all four PEs", "partition a global tensor"):
        if marker not in block_text:
            errors.append(f"get_block_idx contract lacks {marker!r}")

    index = (docs / "intrinsics" / "index.md").read_text(encoding="utf-8")
    for name, path in pages.items():
        if f"({path.name})" not in index:
            errors.append(f"intrinsic index does not link {name}")
    for path in identities.values():
        if f"({path.name})" not in index:
            errors.append(f"intrinsic index does not link {path.name}")
    for marker in ("111", "113", "PTO-ISA-LinxISA 0.57.numbers"):
        if marker not in index:
            errors.append(f"intrinsic index lacks corpus marker {marker!r}")

    shared_contracts = {
        "TLOAD": ("Shared Tile Register Form", "shared tile register as the\ndestination"),
        "TSTORE": ("Shared Tile Register Form", "shared tile version as its source"),
        "TMOV": ("PE-Local and Shared Tile Register Forms", "SharedMove::Insert"),
    }
    for name, markers in shared_contracts.items():
        text = pages[name].read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(f"{pages[name]}: missing shared-register marker {marker!r}")
    for name in ("TMATMUL", "TMATMUL_ACC", "TMATMUL_BIAS", "TMATMUL_MX"):
        text = pages[name].read_text(encoding="utf-8")
        for marker in ("fully defined shared tile", "Four-PE Group Contract"):
            if marker not in text:
                errors.append(f"{pages[name]}: missing shared RHS contract {marker!r}")

    assembly = (docs / "intrinsics" / "assembly.md").read_text(encoding="utf-8")
    for marker in (
        "Assembly provenance",
        "isa/intrinsic/ASSEMBLY_SYNTAX.md",
        "BSTART.TEPL",
        "C.B.IOS",
        "SharedTID",
        "PE_MASK",
        "Shared-Right `TMATMUL`",
    ):
        if marker not in assembly:
            errors.append(f"intrinsic assembly page lacks {marker}")
    if ".reuse" in assembly or "B.IOD" in assembly:
        errors.append("assembly page retains legacy queue-release/dependency syntax")
    for path in pages.values():
        if re.search(r"\bT[A-Z0-9_]+_IMPL\b", path.read_text(encoding="utf-8")):
            errors.append(f"{path}: exposes a non-public implementation helper")


def check_model(docs: Path, errors: list[str]) -> None:
    required = {
        "model/programming.md": ("four processing elements", "same\nentry and initial program counter"),
        "model/group-execution.md": ("get_thread_idx()", "get_block_idx()"),
        "model/shared-tile-registers.md": ("S[0]", "Shared-Right Matrix Multiply"),
        "model/execution.md": ("Tile-ID", "execute in either order"),
        "tutorials/group-programming.md": (
            "Four-PE Grouped Matrix Multiply",
            "Four-PE Reduce-to-One",
            "All-Reduce",
            "Neighbor Exchange",
        ),
    }
    for relative, markers in required.items():
        path = docs / relative
        if not path.is_file():
            errors.append(f"missing programming-model page: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(f"{relative}: missing model marker {marker!r}")

    for removed in ("events.md", "frontends.md", "multicore.md"):
        if (docs / "model" / removed).exists():
            errors.append(f"removed programming-model page still exists: model/{removed}")


def check_catalog(root: Path, docs: Path, errors: list[str]) -> None:
    generator = load_generator(root)
    actual = generator.inventory(root)
    catalog_path = docs / "benchmarks" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = catalog["builds"]
    sources = {item.source for item in actual}
    families = {item.family for item in actual}
    if (len(actual), len(sources), len(families)) != (62, 29, 14):
        errors.append(
            "one-level inventory changed: expected 62 builds/29 implementations/14 "
            f"families, found {len(actual)}/{len(sources)}/{len(families)}"
        )
    if catalog.get("schema") != 3:
        errors.append(f"catalog schema is {catalog.get('schema')}, expected 3")
    if len(rows) != len(actual):
        errors.append(f"catalog has {len(rows)} builds; manifests have {len(actual)}")
    if catalog.get("source_implementations") != len(sources):
        errors.append("catalog implementation count differs from manifest inventory")
    if len(catalog.get("families", [])) != len(families):
        errors.append("catalog family count differs from manifest inventory")

    actual_keys = {
        (generator.rel(item.manifest, root), item.line, item.command) for item in actual
    }
    catalog_keys = {
        (item["manifest"], item["manifest_line"], item["command"]) for item in rows
    }
    for missing in sorted(actual_keys - catalog_keys):
        errors.append(f"manifest command absent from catalog: {missing}")
    for stale in sorted(catalog_keys - actual_keys):
        errors.append(f"stale catalog command: {stale}")

    canonical = set(instruction_pages(docs))
    _, by_name = generator.file_index(root)
    repository_paths = {
        generator.rel(path, root) for paths in by_name.values() for path in paths
    }
    closure_cache: dict[Path, list[Path]] = {}
    for item in rows:
        source = root / item["source"]
        page = docs / item["page"]
        if item["backend"] != "One-level":
            errors.append(f"catalog publishes non-one-level backend: {item['backend']}")
        if not item["source"].startswith("benchmark/one-level-arch/"):
            errors.append(f"catalog publishes non-one-level source: {item['source']}")
        if not source.is_file():
            errors.append(f"catalog source missing: {item['source']}")
        if not page.is_file():
            errors.append(f"catalog page missing: {item['page']}")
            continue
        text = page.read_text(encoding="utf-8")
        for marker in ("## How the Code Is Written", "## Supported PTO Intrinsics"):
            if marker not in text:
                errors.append(f"{page}: missing {marker}")
        if "```cpp" not in text and "```c" not in text:
            errors.append(f"{page}: missing complete source code")
        if item["source_intrinsics"] and "source-level union" not in text:
            errors.append(f"{page}: missing source-level union disclosure")
        if item.get("intrinsic_surface_scope") != "source-union":
            errors.append(f"{page}: missing source-union intrinsic scope")
        for name in item["source_intrinsics"]:
            if name not in canonical:
                errors.append(f"{page}: unknown canonical intrinsic {name}")

        if source.is_file():
            closure = closure_cache.setdefault(
                source, generator.include_closure(source, root, by_name)
            )
            published = [
                path
                for path in closure
                if generator.rel(path, root).startswith("benchmark/one-level-arch/")
            ]
            for closure_path in published:
                closure_name = generator.rel(closure_path, root)
                if f"`{closure_name}`" not in text:
                    errors.append(f"{page}: closure omits {closure_name}")
                closure_text = closure_path.read_text(encoding="utf-8", errors="replace")
                for target in re.findall(
                    r'^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]',
                    closure_text,
                    re.MULTILINE,
                ):
                    if (
                        any(path == target or path.endswith("/" + target) for path in repository_paths)
                        and generator.resolve_include(closure_path, target, root, by_name) is None
                    ):
                        errors.append(
                            f"{page}: unresolved repository include {target} from {closure_name}"
                        )

    control_rows = [item for item in rows if item["testcase"] == "hashtable_lookup_simd"]
    if len(control_rows) != 6:
        errors.append(f"control loop expansion produced {len(control_rows)} builds; expected 6")

    readme = (root / "README.md").read_text(encoding="utf-8")
    section = re.search(
        r"<!-- BENCHMARK-CATALOG:START -->(.*?)<!-- BENCHMARK-CATALOG:END -->",
        readme,
        re.DOTALL,
    )
    if not section:
        errors.append("README benchmark catalog markers are missing")
    else:
        for name in sorted({item.testcase for item in actual}):
            if f"`{name}`" not in section.group(1):
                errors.append(f"README catalog omits TESTCASE={name}")


def resolve_site_target(site: Path, source: Path, link: str) -> Path | None:
    parsed = urlsplit(link)
    if parsed.scheme or parsed.netloc or link.startswith(("mailto:", "javascript:")):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    if path.startswith("/SuperNPUBench/"):
        target = (site / path.removeprefix("/SuperNPUBench/")).resolve()
    elif path.startswith("/"):
        target = (site / path.lstrip("/")).resolve()
    else:
        target = (source.parent / path).resolve()
    try:
        target.relative_to(site.resolve())
    except ValueError:
        return target
    if target.is_dir():
        target = target / "index.html"
    return target


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


def check_removed_surfaces(root: Path, docs: Path, errors: list[str]) -> None:
    if any((docs / "reference").glob("*")):
        errors.append("docs/reference still contains files")
    paths = [root / "README.md", root / "mkdocs.yml", *docs.rglob("*.md")]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"/Users/[^/]+/", text) or "zhoubot" in text:
            errors.append(f"{path.relative_to(root)}: contains a developer-local path")
        if REMOVED_COMPLETION_API.search(text):
            errors.append(f"{path.relative_to(root)}: contains removed completion API terminology")

    mkdocs = (root / "mkdocs.yml").read_text(encoding="utf-8")
    if re.search(r"^\s*- Reference:", mkdocs, re.MULTILINE):
        errors.append("mkdocs navigation still contains a Reference section")
    if REMOVED_WEBSITE_SURFACE.search(mkdocs):
        errors.append("mkdocs navigation exposes a removed architecture surface")
    for path in (docs / "benchmarks").rglob("*.md"):
        if REMOVED_WEBSITE_SURFACE.search(path.read_text(encoding="utf-8", errors="replace")):
            errors.append(f"{path.relative_to(root)}: exposes a non-one-level benchmark surface")


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
    from verify_pto_kernel_migration import check_migration

    check_migration(root, errors)
    check_catalog(root, docs, errors)
    check_removed_surfaces(root, docs, errors)
    check_site((args.site or root / "site").resolve(), errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"golden manual verification failed with {len(errors)} errors")
    print("Golden manual verified: 113 public intrinsics, 62 one-level builds, "
          "29 implementations, links clean.")


if __name__ == "__main__":
    main()
