#!/usr/bin/env python3
"""Embed a validated prefix of a QSMLA FP16 input as a Linx object."""

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--bytes", required=True, type=int)
    parser.add_argument(
        "--exact-bytes", action="store_true",
        help="reject inputs whose size differs from --bytes",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if args.bytes <= 0:
        parser.error("--bytes must be positive")
    if not input_path.is_file():
        parser.error(f"input does not exist: {input_path}")
    input_bytes = input_path.stat().st_size
    if args.exact_bytes and input_bytes != args.bytes:
        parser.error(
            f"input size mismatch: {input_path} has {input_bytes} bytes, "
            f"expected exactly {args.bytes}"
        )
    if input_bytes < args.bytes:
        parser.error(
            f"input is too short: {input_path} has {input_path.stat().st_size} "
            f"bytes, need {args.bytes}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assembly = f"""\
.section .qsmla_input,"a",@progbits
.balign 4096
.global _binary_{args.symbol}_start
_binary_{args.symbol}_start:
.incbin \"{input_path}\", 0, {args.bytes}
.global _binary_{args.symbol}_end
_binary_{args.symbol}_end:
.global _binary_{args.symbol}_size
.equ _binary_{args.symbol}_size, .-_binary_{args.symbol}_start
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".s", prefix="qsmla-input-", delete=False
    ) as asm_file:
        asm_file.write(assembly)
        asm_path = Path(asm_file.name)
    try:
        subprocess.run(
            [
                str(args.compiler), "-target", "linx64v5", "-c",
                str(asm_path), "-o", str(args.output),
            ],
            check=True,
        )
    finally:
        asm_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
