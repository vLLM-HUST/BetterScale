"""Diagnostic Python allocation boundaries during an unchanged NPU capture.

No synchronize, peak reset, tensor retention or device operation is added.
Timings from instrumented runs are diagnostic only. Compare the total peak
against the uninstrumented control before using the attribution.
"""
import sys
from pathlib import Path


class CaptureMemoryTrace:
    def __init__(self, roots, enabled=True):
        self.roots = tuple(str(Path(root).resolve()) + '/' for root in roots)
        self.enabled = enabled
        self.reserved = self.peak_reserved = 0
        self.records = []
        self.lines = {}
        self.filenames = {}
        self.peak = 0
        self.allocated = 0

    def __enter__(self):
        self.previous = sys.gettrace()
        if self.enabled:
            sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)
        self.lines.clear()

    def selected(self, filename):
        if filename not in self.filenames:
            path = str(Path(filename).resolve())
            self.filenames[filename] = path.startswith(self.roots)
        return self.filenames[filename]

    def trace(self, frame, event, arg):
        if not self.selected(frame.f_code.co_filename):
            return None
        if event not in ('line', 'return'):
            return self.trace
        import torch
        allocated = torch.npu.memory_allocated()
        peak = torch.npu.max_memory_allocated()
        reserved = torch.npu.memory_reserved()
        peak_reserved = torch.npu.max_memory_reserved()
        key = id(frame)
        previous_line = self.lines.get(key)
        self.lines[key] = frame.f_lineno
        if (peak != self.peak or peak_reserved != self.peak_reserved
                or abs(allocated - self.allocated) >= 32 * 2**20
                or reserved != self.reserved):
            record = dict(file=frame.f_code.co_filename, function=frame.f_code.co_name,
                          event=event, previous_line=previous_line, line=frame.f_lineno,
                          allocated=allocated, peak=peak, new_peak=peak > self.peak,
                          reserved=reserved, peak_reserved=peak_reserved,
                          peak_reset=peak < self.peak or peak_reserved < self.peak_reserved)
            if peak > self.peak:
                record['stack'] = self.stack(frame)
            self.records.append(record)
            self.allocated = allocated
            self.peak = peak
            self.reserved, self.peak_reserved = reserved, peak_reserved
        if event == 'return':
            self.lines.pop(key, None)
        return self.trace

    def stack(self, frame):
        # Do not inspect f_locals: materializing its dictionary can prolong
        # tensor lifetimes and manufacture exactly the peak we are measuring.
        result = []
        for _ in range(10):
            if frame is None:
                break
            if self.selected(frame.f_code.co_filename):
                result.append(dict(file=frame.f_code.co_filename,
                                   function=frame.f_code.co_name, line=frame.f_lineno))
            frame = frame.f_back
        return result
