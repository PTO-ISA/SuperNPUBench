"""Function-scoped native evidence for runtime-shape QLI TopK."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from native_bstart import normalize_disassembly

from check_disassembly import (function_body_exact, ir_function_body_exact,
                               matching_function_bodies)


def direct_calls(body):
    return re.findall(r'(?:C\.)?BSTART\.STD\s+DIRECT,[^\n]*<([^>]+)>', body)


def check_native(disassembly, ir):
    disassembly = normalize_disassembly(disassembly)
    roots = matching_function_bodies(disassembly, "qli_topk_radix_dynamic")
    if len(roots) != 1:
        raise ValueError("expected one dynamic QLI TopK function")
    root, root_body = roots[0]
    if root not in direct_calls(function_body_exact(disassembly, "main")):
        raise ValueError("dynamic QLI TopK function is not called by main")
    reachable, pending = {}, [root]
    while pending:
        symbol = pending.pop()
        if symbol in reachable:
            continue
        body = function_body_exact(disassembly, symbol)
        reachable[symbol] = body
        pending.extend(direct_calls(body))
    combined = "\n".join(reachable.values())
    if "THISTOGRAM" in combined:
        raise ValueError("retired THISTOGRAM appeared in dynamic QLI closure")
    for engine, opcode in (("TLSU", "MGATHER.ADD"),
                           ("TEPL", "TROWSUM")):
        if not re.search(rf"BSTART\.{engine}\s+{re.escape(opcode)},\s*U32\b",
                         combined):
            raise ValueError("missing scoped dynamic QLI operation " + opcode)
    popn = [symbol for symbol in reachable
            if "PopN" in symbol and symbol.endswith(".pto.local")]
    if not popn:
        raise ValueError("no reachable dynamic PopN .pto.local helper")
    for symbol in popn:
        body = reachable[symbol]
        first = re.search(r"BSTART\.(?:TLSU|TEPL)\s+(\w+)", body)
        if first is None or first.group(1) != "TROWARGMAX":
            raise ValueError("PopN helper did not begin with TROWARGMAX")
        for opcode in ("TCMP", "TSEL"):
            if not re.search(rf"BSTART\.TEPL\s+{opcode},\s*U32\b", body):
                raise ValueError("PopN helper missing typed " + opcode)
        helper_ir = ir_function_body_exact(ir, symbol)
        if re.search(r"\b(?:load|store)\s+<\d+ x", helper_ir):
            raise ValueError("PopN helper spilled its Tile payload")
    return {"root": root, "reachable_from_main": True,
            "popn_local_helpers": popn,
            "scope": "called dynamic TopK closure only"}
