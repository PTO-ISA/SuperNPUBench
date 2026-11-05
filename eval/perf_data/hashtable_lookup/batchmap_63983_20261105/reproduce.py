from pathlib import Path
import os
import json
import subprocess

p = Path(__file__).resolve().parent
model = Path(os.environ.get(
    "MODEL", "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-pr1177"))
record = json.loads((p / "command.json").read_text())
cmd = list(record["canonical_perf"]["argv"])
cmd[0] = str(model / "bin" / "gfsim")
cmd[cmd.index("-f") + 1] = str(p / "hash_6144_PE4_batchmap_perf.elf")
with (p / "rerun.log").open("w") as log:
    run = subprocess.run(cmd, cwd=model, stdout=log, stderr=subprocess.STDOUT)
print("exit:", run.returncode)
print("check Total Cycles (expect 155177) and the finisher in", p / "rerun.log")
