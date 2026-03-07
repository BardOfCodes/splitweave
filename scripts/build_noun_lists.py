#!/usr/bin/env python3
"""
Build noun list files from corpora JSON sources with per-source parsing strategies.

Strategies:
  - list_of_strings: data[key] is a list of strings (e.g. cats.json "cats").
  - list_of_objects: data[key] is a list of objects; extract item[item_key] (e.g. crayola "colors"/"color").
  - root_array: JSON root is a list of objects; extract item[item_key] (e.g. dogs-en-de.json "name").
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_config(path: Path):
    """Load config from YAML or JSON. Prefer YAML if extension is .yaml/.yml."""
    text = path.read_text()
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        if not HAS_YAML:
            raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml")
        return yaml.safe_load(text)
    if suf == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config extension: {suf}. Use .yaml, .yml, or .json")


def _get_nested(obj, path: str):
    """Get value at dotted path, e.g. 'data.melee' -> obj['data']['melee']."""
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _get_by_key_or_path(data: dict, key: str):
    """Get value by literal key first; if key contains '.' and not found, try dotted path."""
    if key in data:
        return data[key]
    if "." in key:
        return _get_nested(data, key)
    return None


def extract_list_of_strings(data: dict, key: str) -> list[str]:
    """Strategy: data[key] is a list of strings. Key may be dotted (e.g. data.melee)."""
    raw = _get_by_key_or_path(data, key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if x is not None and str(x).strip()]


def _get_item_val(item: dict, item_key: str):
    """Get value from item; item_key may be dotted (e.g. person.name)."""
    if "." not in item_key:
        return item.get(item_key)
    return _get_nested(item, item_key)


def extract_list_of_objects(data: dict, key: str, item_key: str) -> list[str]:
    """Strategy: data[key] is a list of objects; take item[item_key] for each. item_key may be dotted."""
    raw = _get_by_key_or_path(data, key)
    if raw is None or not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        val = _get_item_val(item, item_key)
        if val is not None:
            s = str(val).strip()
            if s:
                out.append(s)
    return out


def extract_object_keys(data: dict, key: str) -> list[str]:
    """Strategy: data[key] is an object; return its keys as the noun list (e.g. egyptian_gods)."""
    raw = _get_by_key_or_path(data, key)
    if raw is None or not isinstance(raw, dict):
        return []
    return [str(k).strip() for k in raw if k is not None and str(k).strip()]


def extract_object_values_list_of_objects(data: dict, item_key: str) -> list[str]:
    """Strategy: root is a dict; each value that is a list of dicts -> extract item[item_key] from each."""
    if not isinstance(data, dict):
        return []
    out = []
    for v in data.values():
        if not isinstance(v, list):
            continue
        for item in v:
            if not isinstance(item, dict):
                continue
            val = _get_item_val(item, item_key)
            if val is not None:
                s = str(val).strip()
                if s:
                    out.append(s)
    return out


def extract_root_array(data: list, item_key: str) -> list[str]:
    """Strategy: data is a list of objects; take item[item_key] for each. item_key may be dotted."""
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        val = _get_item_val(item, item_key)
        if val is not None:
            s = str(val).strip()
            if s:
                out.append(s)
    return out


def extract_nouns_from_file(
    file_path: Path,
    strategy: str,
    key: str | None = None,
    item_key: str | None = None,
    root_array: bool = False,
) -> list[str]:
    """
    Load JSON from file_path and return list of noun strings using the given strategy.
    Missing file or invalid data returns [] and logs a warning.
    """
    if not file_path.exists():
        print(f"Warning: file not found, skipping: {file_path}", file=sys.stderr)
        return []

    try:
        raw = json.loads(file_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to load {file_path}: {e}", file=sys.stderr)
        return []

    if strategy == "list_of_strings":
        if key is None:
            print(f"Warning: list_of_strings requires 'key' for {file_path}", file=sys.stderr)
            return []
        return extract_list_of_strings(raw, key)

    if strategy == "list_of_objects":
        if key is None or item_key is None:
            print(f"Warning: list_of_objects requires 'key' and 'item_key' for {file_path}", file=sys.stderr)
            return []
        return extract_list_of_objects(raw, key, item_key)

    if strategy == "root_array":
        if item_key is None:
            print(f"Warning: root_array requires 'item_key' for {file_path}", file=sys.stderr)
            return []
        return extract_root_array(raw, item_key)

    if strategy == "object_keys":
        if key is None:
            print(f"Warning: object_keys requires 'key' for {file_path}", file=sys.stderr)
            return []
        return extract_object_keys(raw, key)

    if strategy == "object_values_list_of_objects":
        if item_key is None:
            print(f"Warning: object_values_list_of_objects requires 'item_key' for {file_path}", file=sys.stderr)
            return []
        return extract_object_values_list_of_objects(raw, item_key)

    print(f"Warning: unknown strategy '{strategy}' for {file_path}", file=sys.stderr)
    return []


def resolve_path(base: Path, file_ref: str) -> Path:
    """If file_ref is absolute, return it; else resolve against base."""
    p = Path(file_ref)
    if p.is_absolute():
        return p
    return (base / file_ref).resolve()


def build_one_noun_list(
    output_path: Path,
    sources: list[dict],
    base_path: Path,
    sort: bool = True,
    dedupe: bool = True,
) -> None:
    """Gather nouns from all sources, dedupe/sort, write to output_path."""
    seen: set[str] = set()
    ordered: list[str] = []

    for src in sources:
        file_ref = src.get("file")
        if not file_ref:
            print("Warning: source missing 'file', skipping", file=sys.stderr)
            continue
        strategy = src.get("strategy", "list_of_strings")
        key = src.get("key")
        item_key = src.get("item_key")
        root_array = src.get("root_array", False)

        full_path = resolve_path(base_path, file_ref)
        nouns = extract_nouns_from_file(
            full_path,
            strategy=strategy,
            key=key,
            item_key=item_key,
            root_array=root_array,
        )
        for n in nouns:
            if dedupe:
                norm = n.strip().casefold()
                if norm in seen:
                    continue
                seen.add(norm)
            ordered.append(n)

    if sort:
        ordered.sort(key=lambda s: s.strip().casefold())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")
    print(f"Wrote {len(ordered)} nouns to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build noun list files from corpora JSON with configurable parsing strategies."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "noun_lists_config.example.yaml",
        help="Path to YAML or JSON config (default: noun_lists_config.example.yaml next to script)",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Override base path for resolving relative 'file' paths in config",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort nouns in output",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate nouns across sources",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    base_path = args.base_path
    if base_path is None:
        base_cfg = config.get("base_path")
        if base_cfg:
            p = Path(base_cfg)
            base_path = p if p.is_absolute() else (Path.cwd() / base_cfg)
        else:
            base_path = Path.cwd()
    base_path = base_path.resolve()

    noun_lists = config.get("noun_lists") or []
    if not noun_lists:
        print("No noun_lists defined in config.", file=sys.stderr)
        sys.exit(0)

    for entry in noun_lists:
        output_path = Path(entry.get("output_path", "")).resolve()
        sources = entry.get("sources") or []
        if not output_path:
            print("Warning: entry missing output_path, skipping", file=sys.stderr)
            continue
        build_one_noun_list(
            output_path,
            sources,
            base_path,
            sort=not args.no_sort,
            dedupe=not args.no_dedupe,
        )


if __name__ == "__main__":
    main()
