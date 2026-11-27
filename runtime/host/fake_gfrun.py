#!/usr/bin/env python3
"""Protocol stub for the native host-runtime unit test."""

import os
import sys
import time


print("HSCB KERNEL 1000 ok_kernel", flush=True)
print("HSCB KERNEL 2000 fail_kernel", flush=True)
print("HSCB HOST OPEN workers=1", flush=True)
next_address = 0x100000000
free_ranges = []
firmware = sys.argv[sys.argv.index("-f") + 1]
launch_count = 0

for line in sys.stdin:
    fields = line.split()
    command, request = fields[:2]
    if command == "ALLOC":
        size = int(fields[2])
        if free_ranges:
            address = free_ranges.pop(0)
        else:
            address = next_address
            next_address += 4096
        print(f"HSCB HOST ALLOC {request} status=OK address={address} size={size}", flush=True)
    elif command == "FREE":
        free_ranges.append(int(fields[2]))
        print(f"HSCB HOST FREE {request} status=OK address={fields[2]}", flush=True)
    elif command == "INIT":
        print(f"HSCB HOST INITIALIZING {request} status=OK", flush=True)
        print("HSCB HOST READY workers=1", flush=True)
    elif command == "WRITE":
        size = len(open(" ".join(fields[3:]), "rb").read())
        print(f"HSCB HOST WRITE {request} status=OK size={size}", flush=True)
    elif command == "LAUNCH":
        block_count = fields[4]
        print(f"HSCB HOST SUBMITTED {request} status=OK blocks={block_count}", flush=True)
        if fields[2] == "4096":
            if launch_count == 0:
                deadline = time.monotonic() + 5
                while not os.path.exists(firmware + ".submitted"):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("second enqueue marker was not published")
                    time.sleep(0.01)
            status = 0
        else:
            status = 7
        print(f"HSCB HOST COMPLETE {request} status={status} blocks={block_count}", flush=True)
        launch_count += 1
    elif command == "QUIT":
        print(f"HSCB HOST BYE {request} status=OK", flush=True)
        break
