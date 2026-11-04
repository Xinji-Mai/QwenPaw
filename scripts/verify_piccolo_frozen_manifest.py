"""Run the production runtime loader in a minimal PyInstaller bundle."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


root = Path.cwd()
output = root / "frozen-manifest-results"
output.mkdir(exist_ok=True)
source = root / "src/qwenpaw"
spec = root / "scripts/pack-tauri/qwenpaw.spec"
selected = []
for node in ast.parse(spec.read_text(encoding="utf-8")).body:
    if "runtime-release.json" in ast.get_source_segment(
        spec.read_text(encoding="utf-8"), node
    ):
        selected.append(node)
assert len(selected) == 1, "Expected exactly one runtime manifest data entry"
namespace = {"SRC": source, "datas": []}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(spec), "exec"), namespace)
data = namespace["datas"]
assert len(data) == 1
manifest_source, destination = data[0]
expected = json.loads(Path(manifest_source).read_text(encoding="utf-8"))
assert destination == "qwenpaw/local_models"

fixture = output / "fixture"
package = fixture / "qwenpaw/local_models"
package.mkdir(parents=True, exist_ok=True)
(fixture / "qwenpaw/__init__.py").touch()
(package / "__init__.py").touch()
shutil.copy2(source / "local_models/runtime_release.py", package / "runtime_release.py")
probe = output / "probe.py"
probe.write_text(
    "import json\nfrom qwenpaw.local_models.runtime_release import RELEASE\n"
    "print(json.dumps(RELEASE, sort_keys=True))\n",
    encoding="utf-8",
)
reports = {}
for stage in ("before", "after"):
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--onedir", "--name", "runtime-manifest-probe", "--paths", str(fixture),
        "--distpath", str(output / f"dist-{stage}"),
        "--workpath", str(output / f"work-{stage}"),
        "--specpath", str(output / f"spec-{stage}"),
    ]
    if stage == "after":
        command += ["--add-data", f"{manifest_source}:{destination}"]
    command.append(str(probe))
    with (output / f"build-{stage}.log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    executable = output / f"dist-{stage}/runtime-manifest-probe/runtime-manifest-probe"
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as run_directory:
        result = subprocess.run(
            [str(executable)], cwd=run_directory, env=environment,
            capture_output=True, text=True, timeout=60,
        )
    (output / f"run-{stage}.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    if stage == "before":
        assert result.returncode != 0
        assert "FileNotFoundError" in result.stderr
        assert "runtime-release.json" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == expected
    reports[stage] = {"returncode": result.returncode, "verified": True}
reports.update({
    "platform": sys.platform,
    "python": sys.version,
    "release": expected["release"],
    "targets": len(expected["targets"]),
    "manifest_sha256": hashlib.sha256(Path(manifest_source).read_bytes()).hexdigest(),
    "loader_sha256": hashlib.sha256((package / "runtime_release.py").read_bytes()).hexdigest(),
})
(output / "verification.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
print(json.dumps(reports, indent=2))
