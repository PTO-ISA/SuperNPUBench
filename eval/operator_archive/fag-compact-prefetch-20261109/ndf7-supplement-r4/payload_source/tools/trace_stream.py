"""Bounded JSON trace reader; only one event and small metadata are decoded at once."""
import json
import gzip
from pathlib import Path


class TraceReader:
    def __init__(self, path, chunk_size=65536, max_value_chars=4 * 1024 * 1024):
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.max_value_chars = max_value_chars
        self.metadata = {}
        self.is_array = False

    def __iter__(self):
        decoder = json.JSONDecoder()
        with (gzip.open(self.path, 'rt', encoding='utf-8') if self.path.suffix == '.gz' else self.path.open(encoding='utf-8')) as handle:
            buf = ''
            pos = 0
            eof = False

            def fill():
                nonlocal buf, pos, eof
                buf = buf[pos:]
                pos = 0
                part = handle.read(self.chunk_size)
                eof = not part
                buf += part
                if len(buf) > self.max_value_chars + self.chunk_size:
                    raise ValueError('trace JSON value exceeds bounded reader limit')

            def peek():
                nonlocal pos
                while True:
                    while pos < len(buf) and buf[pos].isspace():
                        pos += 1
                    if pos < len(buf):
                        return buf[pos]
                    if eof:
                        return ''
                    fill()

            def expect(char):
                nonlocal pos
                if peek() != char:
                    raise ValueError(f'expected {char!r} in trace JSON')
                pos += 1

            def value():
                nonlocal pos
                peek()
                while True:
                    try:
                        result, end = decoder.raw_decode(buf, pos)
                        # Numbers may continue across a read boundary.
                        if end == len(buf) and not eof:
                            fill()
                            continue
                        if (isinstance(result, (int, float)) and not isinstance(result, bool)
                                and end < len(buf) and buf[end] not in ' \t\r\n,]}:'):
                            # raw_decode accepts a numeric prefix such as 1 in
                            # an incomplete 1. or 1e+ split across read blocks.
                            if not eof:
                                fill()
                                continue
                            raise ValueError('invalid or incomplete JSON number')
                        pos = end
                        return result
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        fill()

            def events():
                expect('[')
                if peek() == ']':
                    expect(']')
                    return
                while True:
                    event = value()
                    if not isinstance(event, dict):
                        raise ValueError('trace event must be an object')
                    yield event
                    if peek() == ']':
                        expect(']')
                        return
                    expect(',')

            self.metadata = {}
            self.is_array = peek() == '['
            if self.is_array:
                yield from events()
            else:
                expect('{')
                found = False
                if peek() != '}':
                    while True:
                        key = value()
                        if not isinstance(key, str):
                            raise ValueError('trace root key must be a string')
                        expect(':')
                        if key == 'traceEvents':
                            if found:
                                raise ValueError('duplicate traceEvents')
                            found = True
                            yield from events()
                        else:
                            self.metadata[key] = value()
                        if peek() == '}':
                            break
                        expect(',')
                expect('}')
                if not found:
                    raise ValueError('missing traceEvents')
            if peek():
                raise ValueError('trailing trace JSON data')


def strip_counters(src, dst):
    reader = TraceReader(src)
    total = kept = 0
    with Path(dst).open('w', encoding='utf-8') as out:
        # Determine root kind without consuming or retaining events.
        iterator = iter(reader)
        first = next(iterator, None)
        out.write('[' if reader.is_array else '{"traceEvents":[')
        def emit(event):
            nonlocal total, kept
            total += 1
            if event.get('ph') != 'C':
                if kept:
                    out.write(',')
                json.dump(event, out, ensure_ascii=False, separators=(',', ':'))
                kept += 1
        if first is not None:
            emit(first)
        for event in iterator:
            emit(event)
        out.write(']')
        if not reader.is_array:
            for key, val in reader.metadata.items():
                out.write(',')
                json.dump(key, out, ensure_ascii=False)
                out.write(':')
                json.dump(val, out, ensure_ascii=False)
            out.write('}')
        out.write('\n')
    return total, kept
