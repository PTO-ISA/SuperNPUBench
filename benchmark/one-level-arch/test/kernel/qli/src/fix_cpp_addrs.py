import re, sys, subprocess

cpp_path = sys.argv[1]
elf_path = sys.argv[2]
nm_path = sys.argv[3]

# Get symbol addresses from ELF
result = subprocess.run([nm_path, elf_path], capture_output=True, text=True)
addrs = {}
for line in result.stdout.split('\n'):
    for name in ['srcq', 'srck', 'srcw', 'srcsq', 'srcsk']:
        if f'_binary_{name}_data_start' in line:
            parts = line.strip().split()
            addrs[name] = parts[0]
            break

with open(cpp_path, 'r') as f:
    content = f.read()

# Replace each SRC*_ADDR with exact new value
name_map = {'srcq': 'SRCQ', 'srck': 'SRCK', 'srcw': 'SRCW', 'srcsq': 'SRCSQ', 'srcsk': 'SRCSK'}
for name, prefix in name_map.items():
    if name in addrs:
        addr = addrs[name]
        new_val = f'0x{addr:016s}ULL'
        pattern = rf'#define {prefix}_ADDR\s+0x[0-9a-fA-F]+ULL'
        replacement = f'#define {prefix}_ADDR  {new_val}'
        content = re.sub(pattern, replacement, content)
        print(f'  {prefix}_ADDR = {new_val}')

with open(cpp_path, 'w') as f:
    f.write(content)
print('Addresses updated')
