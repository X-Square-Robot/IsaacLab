# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script sets up the VS Code ty (Python type checker) configuration for the Isaac Lab project.

It dynamically discovers extra paths from IsaacSim extensions and IsaacLab source packages,
then updates ``ty.configuration.environment.python`` and ``ty.configuration.environment.extra-paths``
in ``.vscode/settings.json``.
"""

import os
import pathlib
import re
import subprocess
import sys

ISAACLAB_DIR = pathlib.Path(__file__).parents[2]
"""Path to the Isaac Lab directory."""

# Try to find IsaacSim dir
_isaacsim_probe = subprocess.run(
    [sys.executable, "-c", "import isaacsim; import os; print(os.environ.get('ISAAC_PATH', ''))"],
    capture_output=True,
    text=True,
    check=False,
    stdin=subprocess.DEVNULL,
)
if _isaacsim_probe.returncode == 0 and _isaacsim_probe.stdout.strip():
    isaacsim_dir = _isaacsim_probe.stdout.strip()
else:
    isaacsim_dir = os.path.join(ISAACLAB_DIR, "_isaac_sim")

if not os.path.exists(isaacsim_dir):
    print(
        f"[WARN] Could not find the isaac-sim directory: {isaacsim_dir}."
        "\n\tIsaac Sim does not appear to be installed. VS Code settings will be generated"
        "\n\twithout Isaac Sim extra paths."
    )
    isaacsim_dir = ""

ISAACSIM_DIR = isaacsim_dir
"""Path to the isaac-sim directory."""


def find_extra_paths(isaaclab_dir: pathlib.Path, isaacsim_dir: str) -> list[str]:
    """Scan IsaacSim and source dirs, return list of relative paths.

    Args:
        isaaclab_dir: Path to the Isaac Lab root directory.
        isaacsim_dir: Path to the IsaacSim directory (empty string if not found).

    Returns:
        List of paths relative to ``isaaclab_dir`` with ``./`` prefix.
    """
    paths: list[str] = []

    if isaacsim_dir and os.path.isdir(isaacsim_dir):
        rel_sim = os.path.relpath(isaacsim_dir, isaaclab_dir).replace("\\", "/")

        # Directories where each subdirectory is an extension
        ext_dirs = ["exts", "extscache", "extsDeprecated", "kit/extscore"]
        for ext_dir in ext_dirs:
            full_dir = os.path.join(isaacsim_dir, ext_dir)
            if not os.path.isdir(full_dir):
                continue
            for entry in sorted(os.listdir(full_dir)):
                entry_path = os.path.join(full_dir, entry)
                if os.path.isdir(entry_path):
                    paths.append(f"./{rel_sim}/{ext_dir}/{entry}")
                    # Also add pip_prebundle if it exists
                    pip_prebundle = os.path.join(entry_path, "pip_prebundle")
                    if os.path.isdir(pip_prebundle):
                        paths.append(f"./{rel_sim}/{ext_dir}/{entry}/pip_prebundle")

        # Fixed paths
        fixed_paths = [
            "kit/kernel/py",
            "kit/plugins/bindings-python",
            "kit/python/lib/python3.12",
            "kit/python/lib/python3.12/site-packages",
        ]
        for p in fixed_paths:
            full_path = os.path.join(isaacsim_dir, p)
            if os.path.isdir(full_path):
                paths.append(f"./{rel_sim}/{p}")

    # Add IsaacLab source packages
    source_dir = os.path.join(isaaclab_dir, "source")
    if os.path.isdir(source_dir):
        for entry in sorted(os.listdir(source_dir)):
            if os.path.isdir(os.path.join(source_dir, entry)):
                paths.append(f"./source/{entry}")

    return paths


def update_ty_configuration(settings_content: str, python_path: str, extra_paths: list[str]) -> str:
    """Insert or replace the ty.configuration block in settings JSON string.

    Args:
        settings_content: The current contents of settings.json.
        python_path: Relative path to the Python interpreter.
        extra_paths: List of relative extra paths for ty.

    Returns:
        Updated settings.json content.
    """
    # Build the ty.configuration block
    paths_str = ",\n".join(f'                "{p}"' for p in extra_paths)
    ty_block = (
        '"ty.configuration": {\n'
        '        "environment": {\n'
        f'            "python": "{python_path}",\n'
        '            "extra-paths": [\n'
        f"{paths_str}\n"
        "            ]\n"
        "        }\n"
        "    }"
    )

    # Try to replace existing ty.configuration block
    pattern = r'"ty\.configuration"\s*:\s*\{[^{}]*\{[^{}]*\[.*?\][^{}]*\}[^{}]*\}'
    match = re.search(pattern, settings_content, flags=re.DOTALL)
    if match:
        settings_content = settings_content[: match.start()] + ty_block + settings_content[match.end() :]
    else:
        # Insert before the closing }
        # Find the last } in the file
        last_brace = settings_content.rfind("}")
        if last_brace == -1:
            raise ValueError("Could not find closing } in settings.json")
        # Insert before the last }, after the previous entry
        # We need to add a comma after the previous entry if not present
        before = settings_content[:last_brace].rstrip()
        if before.endswith(","):
            settings_content = before + "\n    " + ty_block + "\n}\n"
        else:
            settings_content = before + ",\n    " + ty_block + "\n}\n"

    return settings_content


def main():
    # Paths
    settings_path = os.path.join(ISAACLAB_DIR, ".vscode", "settings.json")

    if not os.path.exists(settings_path):
        print(f"[ERROR] settings.json not found at {settings_path}. Run setup_vscode.py first.")
        sys.exit(1)

    # Discover extra paths
    extra_paths = find_extra_paths(ISAACLAB_DIR, ISAACSIM_DIR)
    # Point ty at the interpreter currently running this script (i.e. the one
    # `isaaclab.sh -p` resolved), rather than a hard-coded env path.
    python_path = "./" + os.path.relpath(sys.executable, ISAACLAB_DIR).replace("\\", "/")

    # Read existing settings
    with open(settings_path) as f:
        settings_content = f.read()

    # Update ty.configuration
    settings_content = update_ty_configuration(settings_content, python_path, extra_paths)

    # Write back
    with open(settings_path, "w") as f:
        f.write(settings_content)

    print(f"[INFO] Updated ty.configuration in {settings_path}")
    print(f"[INFO] Python path: {python_path}")
    print(f"[INFO] Extra paths: {len(extra_paths)} entries")


if __name__ == "__main__":
    main()
