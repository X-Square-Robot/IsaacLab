# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate/update .pyi stubs for modules using lazy_export() + __getattr__ forwarding.

For each target __init__.py, extracts forwarded frozensets and their target modules
from __getattr__, then either:
  - Generates a complete .pyi (for modules with no hand-written direct imports)
  - Patches the forwarded section of an existing .pyi (preserving hand-written parts)

Usage:
    python tools/generate_lazy_stubs.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

HEADER = """\
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""

# Modules to process. "mode" controls behavior:
#   "full" = overwrite entire .pyi (module only has lazy_export + forwards)
#   "patch" = replace only the "# Forwarded via __getattr__" sections + update __all__
TARGETS = [
    {
        "path": "source/isaaclab/isaaclab/sim/schemas/__init__.py",
        "mode": "full",
        "local_imports": [
            {
                "module": ".schemas",
                "source": "source/isaaclab/isaaclab/sim/schemas/schemas.py",
                "include_constants": True,
                "include_functions": True,
            },
            {
                "module": ".schemas_actuators",
                "source": "source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py",
                "names": ["define_actuator_properties"],
            },
            {
                "module": ".schemas_cfg",
                "source": "source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py",
                "include_classes": True,
            },
        ],
    },
    {
        "path": "source/isaaclab/isaaclab/sim/__init__.py",
        "mode": "patch",
        "local_imports": [
            {
                "module": ".schemas",
                "source": "source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py",
                "include_classes": True,
            },
        ],
    },
]


def extract_frozensets(source: str):
    """Extract all module-level _*FORWARD* frozenset({...}) assignments."""
    tree = ast.parse(source)
    result = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name and name.startswith("_") and "FORWARD" in name:
                if isinstance(node.value, ast.Call) and node.value.args:
                    arg = node.value.args[0]
                    if isinstance(arg, ast.Set):
                        names = [elt.value for elt in arg.elts if isinstance(elt, ast.Constant)]
                        result[name] = sorted(names)
    return result


def extract_getattr_mapping(source: str) -> dict[str, str]:
    """Map frozenset variable names to their import target module.

    Parses:
        if name in _VAR:
            try:
                from some.module import thing as _alias
    """
    mapping = {}
    pattern = re.compile(
        r"if name in (\w+):\s*\n"
        r"\s*try:\s*\n"
        r"\s*from ([\w.]+) import (\w+) as \w+",
    )
    for m in pattern.finditer(source):
        var_name = m.group(1)
        module_path = f"{m.group(2)}.{m.group(3)}"
        mapping[var_name] = module_path
    return mapping


def extract_combined_frozensets(source: str):
    """Find _X = _A | _B patterns."""
    pattern = re.compile(r"(?m)^(\w+)\s*=\s*((?:_\w+\s*\|\s*)+_\w+)\s*$")
    result = {}
    for m in pattern.finditer(source):
        var = m.group(1)
        components = [c.strip() for c in m.group(2).split("|")]
        result[var] = components
    return result


def extract_direct_imports(source: str) -> list[tuple[str, list[str]]]:
    """Extract from .xxx import (...) statements."""
    tree = ast.parse(source)
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in node.names if alias.name != "*"]
            if names and "lazy_export" not in names:
                imports.append((node.module, sorted(names)))
    return imports


def extract_local_imports(target: dict) -> list[tuple[str, list[str]]]:
    """Extract configured local imports from source files.

    This is needed for modules whose runtime ``__init__.py`` gets its local
    lazy exports from the ``.pyi`` stub itself. In those cases, the stub
    generator cannot infer direct imports from ``__init__.py`` alone.
    """
    imports = []
    for cfg in target.get("local_imports", []):
        source_path = REPO_ROOT / cfg["source"]
        tree = ast.parse(source_path.read_text())
        # Explicit allowlist: names listed here are always included (validated to exist
        # as a top-level definition), independent of the suffix/include_* harvest flags.
        explicit = set(cfg.get("names", []))
        defined = {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = explicit - defined
        if missing:
            raise ValueError(f"{cfg['source']}: configured names not found: {sorted(missing)}")
        names = list(explicit)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                suffix = cfg.get("class_suffix")
                if (suffix and node.name.endswith(suffix)) or (
                    cfg.get("include_classes") and not node.name.startswith("_")
                ):
                    names.append(node.name)
            elif cfg.get("include_functions") and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    names.append(node.name)
            elif cfg.get("include_constants") and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target_node in targets:
                    if (
                        isinstance(target_node, ast.Name)
                        and target_node.id.isupper()
                        and not target_node.id.startswith("_")
                    ):
                        names.append(target_node.id)
        imports.append((cfg["module"], sorted(names)))
    return imports


def render_import_block(module: str, names: list[str]) -> str:
    """Render a parenthesized import block."""
    lines = [f"from {module} import ("]
    for name in sorted(set(names)):
        lines.append(f"    {name},")
    lines.append(")")
    return "\n".join(lines)


def ensure_import_blocks(source: str, imports: list[tuple[str, list[str]]]) -> str:
    """Ensure existing ``from module import (...)`` blocks contain generated names."""
    for module, names in imports:
        pattern = re.compile(rf"(?s)from {re.escape(module)} import \(\n(?P<body>.*?)\n\)")
        match = pattern.search(source)
        if match:
            existing_names = re.findall(r"(?m)^\s+(\w+)(?:\s+as\s+\w+)?,", match.group("body"))
            replacement = render_import_block(module, sorted(set(existing_names) | set(names)))
            source = source[: match.start()] + replacement + source[match.end() :]
        else:
            source = source.rstrip() + "\n" + render_import_block(module, names) + "\n"
    return source


def build_forwarded_imports(source: str) -> tuple[list[str], list[str]]:
    """Build import lines and name list for all forwarded symbols."""
    frozensets = extract_frozensets(source)
    getattr_map = extract_getattr_mapping(source)
    combined = extract_combined_frozensets(source)

    # Expand combined frozensets
    for combined_var, components in combined.items():
        if combined_var in getattr_map:
            for comp in components:
                if comp not in getattr_map:
                    getattr_map[comp] = getattr_map[combined_var]

    # Group by target module
    target_groups = {}
    for var_name, target_module in getattr_map.items():
        if var_name in frozensets:
            target_groups.setdefault(target_module, []).extend(frozensets[var_name])

    lines: list[str] = []
    all_names: list[str] = []

    for target_module, names in sorted(target_groups.items()):
        names = sorted(set(names))
        all_names.extend(names)
        parts = target_module.rsplit(".", 1)
        mod_path = f"{parts[0]}.{parts[1]}"
        lines.append(f"# Forwarded via __getattr__ → {target_module}")
        lines.append(f"from {mod_path} import (")
        for n in names:
            lines.append(f"    {n} as {n},")
        lines.append(")")
        lines.append("")

    return lines, sorted(set(all_names))


def generate_full(init_path: Path, target: dict) -> str:
    """Generate complete .pyi for a module with only lazy_export + forwards."""
    source = init_path.read_text()
    direct_imports = extract_direct_imports(source) + extract_local_imports(target)
    forwarded_lines, forwarded_names = build_forwarded_imports(source)

    all_names: list[str] = []

    import_lines: list[str] = []
    for module, names in direct_imports:
        all_names.extend(names)
        import_lines.append(f"from {module} import (")
        for n in names:
            import_lines.append(f"    {n},")
        import_lines.append(")")
        import_lines.append("")

    all_names.extend(forwarded_names)
    all_names = sorted(set(all_names))

    all_block = "__all__ = [\n"
    for n in all_names:
        all_block += f'    "{n}",\n'
    all_block += "]\n"

    return (HEADER + all_block + "\n" + "\n".join(import_lines) + "\n".join(forwarded_lines)).rstrip() + "\n"


def patch_forwarded(init_path: Path, target: dict) -> str:
    """Patch only the forwarded section of an existing .pyi, preserving direct imports."""
    pyi_path = init_path.with_suffix(".pyi")
    if not pyi_path.exists():
        raise FileNotFoundError(f"{pyi_path} must exist for patch mode")

    source = init_path.read_text()
    existing = pyi_path.read_text()
    forwarded_lines, forwarded_names = build_forwarded_imports(source)
    local_imports = extract_local_imports(target)
    local_names = [name for _, names in local_imports for name in names]

    # Remove old forwarded blocks line-by-line
    # Old style: "# Forwarded..." + "X = ..." lines
    # New style: "# Forwarded..." + "from X import (" + indented lines + ")"
    out_lines: list[str] = []
    lines = existing.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"# Forwarded (via __getattr__|to .+ via __getattr__)", line):
            i += 1
            while i < len(lines):
                current_line = lines[i]
                # Skip: "X = ...", blank, "from ext_pkg... import (", indented, ")"
                if re.match(r"(\w+ = \.\.\.\s*$|\s*$)", current_line):
                    i += 1
                elif re.match(r"from (isaaclab_physx|isaaclab_newton)\.", current_line):
                    # Skip entire import block
                    i += 1
                    while i < len(lines) and not lines[i].startswith(")"):
                        i += 1
                    if i < len(lines):
                        i += 1  # skip ")"
                else:
                    break
        else:
            out_lines.append(line)
            i += 1

    cleaned = "".join(out_lines)

    cleaned = ensure_import_blocks(cleaned, local_imports)

    # Update __all__ to include forwarded and generated local names.
    all_match = re.search(r"(?s)__all__ = \[(.*?)\]", cleaned)
    if all_match:
        existing_names = set(re.findall(r'"(\w+)"', all_match.group(1)))
        all_names = sorted(existing_names | set(forwarded_names) | set(local_names))
        new_all = "__all__ = [\n"
        for n in all_names:
            new_all += f'    "{n}",\n'
        new_all += "]"
        cleaned = cleaned[: all_match.start()] + new_all + cleaned[all_match.end() :]

    # Append forwarded imports at end
    cleaned = (cleaned.rstrip() + "\n\n" + "\n".join(forwarded_lines)).rstrip() + "\n"

    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether generated .pyi stubs are up to date without writing files.",
    )
    args = parser.parse_args()

    stale_paths = []
    for target in TARGETS:
        init_path = REPO_ROOT / target["path"]
        if not init_path.exists():
            print(f"[SKIP] {init_path} not found")
            continue

        if target["mode"] == "full":
            content = generate_full(init_path, target)
        else:
            content = patch_forwarded(init_path, target)

        pyi_path = init_path.with_suffix(".pyi")
        relative_path = pyi_path.relative_to(REPO_ROOT)
        if args.check:
            if not pyi_path.exists() or pyi_path.read_text() != content:
                stale_paths.append(relative_path)
                print(f"[STALE] {relative_path}")
            else:
                print(f"[OK] {relative_path} ({target['mode']})")
        else:
            pyi_path.write_text(content)
            print(f"[OK] {relative_path} ({target['mode']})")

    if stale_paths:
        print("Generated stubs are stale; run this script without --check.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
