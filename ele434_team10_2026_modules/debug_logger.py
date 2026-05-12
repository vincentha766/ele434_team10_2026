"""
Lightweight CSV + event logger for navigation runs.

Goal: capture enough per-run data offline that hardware behavior can be
reproduced or diagnosed without rerunning on the real robot. Each run gets
its own timestamped folder under EL434_LOG_DIR (default: ./.tmp_logs/).

Files written per run:
  trace.csv     per-iteration numeric state (time + caller-defined fields)
  events.log    human-readable events (scored / stuck / defer / brake / ...)
  params.txt    parameters captured at start
  summary.txt   end-of-run totals (also written on KeyboardInterrupt)
  snap_*.txt    optional snapshots (e.g. lidar ranges on stuck/brake)

The logger is best-effort: every public method swallows IO errors so a
disk hiccup never takes the robot down mid-run.
"""

import csv
import os
import time
from datetime import datetime


class RunLogger:
    def __init__(self, run_name, trace_fields, params=None, log_dir=None):
        base = log_dir or os.environ.get('EL434_LOG_DIR', '.tmp_logs')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = os.path.join(base, f'{run_name}_{ts}')
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception:
            self.dir = os.path.join('/tmp', f'{run_name}_{ts}')
            os.makedirs(self.dir, exist_ok=True)

        self.t0 = time.time()
        self._trace_fields = list(trace_fields)
        self._counters = {}
        self._snap_n = 0

        self._trace_f = open(os.path.join(self.dir, 'trace.csv'), 'w', newline='')
        self._trace_w = csv.writer(self._trace_f)
        self._trace_w.writerow(['t'] + self._trace_fields)

        self._events_f = open(os.path.join(self.dir, 'events.log'), 'w')
        self._events_f.write(f'# run={run_name} started={ts}\n')
        self._events_f.flush()

        if params:
            try:
                with open(os.path.join(self.dir, 'params.txt'), 'w') as f:
                    for k, v in params.items():
                        f.write(f'{k}={v}\n')
            except Exception:
                pass

    def trace(self, **kwargs):
        try:
            row = [f'{time.time() - self.t0:.3f}']
            for k in self._trace_fields:
                v = kwargs.get(k, '')
                if isinstance(v, float):
                    row.append(f'{v:.4f}')
                elif isinstance(v, bool):
                    row.append('1' if v else '0')
                else:
                    row.append(v)
            self._trace_w.writerow(row)
            self._trace_f.flush()
        except Exception:
            pass

    def event(self, kind, message=''):
        try:
            t = time.time() - self.t0
            self._events_f.write(f'{t:7.2f}  {kind:<14}  {message}\n')
            self._events_f.flush()
            self._counters[kind] = self._counters.get(kind, 0) + 1
        except Exception:
            pass

    def snapshot(self, tag, content):
        """Write an arbitrary text snapshot (e.g. lidar ranges on a stuck event)."""
        try:
            self._snap_n += 1
            name = f'snap_{self._snap_n:03d}_{tag}.txt'
            path = os.path.join(self.dir, name)
            with open(path, 'w') as f:
                t = time.time() - self.t0
                f.write(f'# t={t:.2f} tag={tag}\n')
                if isinstance(content, str):
                    f.write(content)
                    if not content.endswith('\n'):
                        f.write('\n')
                else:
                    for line in content:
                        f.write(str(line) + '\n')
        except Exception:
            pass

    def close(self, summary=None):
        try:
            with open(os.path.join(self.dir, 'summary.txt'), 'w') as f:
                f.write(f'duration_s={time.time() - self.t0:.2f}\n')
                for k, v in self._counters.items():
                    f.write(f'event_{k}={v}\n')
                if summary:
                    for k, v in summary.items():
                        f.write(f'{k}={v}\n')
        except Exception:
            pass
        for fh in (getattr(self, '_trace_f', None), getattr(self, '_events_f', None)):
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
