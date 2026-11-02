from pathlib import Path
import os,json,subprocess
p=Path(__file__).resolve().parent
model=Path(os.environ.get('MODEL','/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel'))
record=json.loads((p/'command.json').read_text());cmd=record['argv'];cmd[0]=str(model/'bin/gfsim');cmd[cmd.index('-f')+1]=str(p/'hash_6144_PE4.elf')
for flag,name in [('--swimfile','rerun.swimlane.json'),('--pipefile','rerun.pipeview.log')]:cmd[cmd.index(flag)+1]=str(p/name)
with (p/'rerun.log').open('w') as log:r=subprocess.run(cmd,cwd=model,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Check all four PE finisher values (0x5555), not exit status alone:',p/'rerun.log')
