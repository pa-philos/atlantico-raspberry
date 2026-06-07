"""Centralized logging setup for Atlantico RPi.

This module exposes a single idempotent helper `setup_logging()` which ensures
there is a stdout StreamHandler and (optionally) a FileHandler writing to
`run/logs/device.log`. The function is safe to call multiple times.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

LOG_PATH = os.environ.get("ATLANTICO_DEVICE_LOG", os.path.join("run", "logs", "device.log"))

_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")


def setup_logging(force_file: bool = False) -> None:
    """Idempotently add a stdout StreamHandler and optional FileHandler.

    If force_file is True or the env var ATLANTICO_DEVICE_CREATE_FILE is set to
    '1', ensure a FileHandler is added writing to LOG_PATH.
    """

    if '--connect' in sys.argv or (__name__ == '__main__'):
        # Prefer creating the file handler for CLI runs
        os.environ.setdefault('ATLANTICO_DEVICE_CREATE_FILE', '1')

    rl = logging.getLogger()
    # Clear existing handlers to prevent duplicates from libraries like absl
    for h in rl.handlers[:]:
        rl.removeHandler(h)
        
    rl.setLevel(logging.INFO)

    want_file = force_file or os.environ.get('ATLANTICO_DEVICE_CREATE_FILE', '0') == '1'
    abs_log_path = os.path.abspath(LOG_PATH) if want_file else None
    
    # If we are writing to a file, and that file is likely where stdout is already 
    # going (via shell redirection), we should be careful about adding BOTH 
    # a StreamHandler and a FileHandler.
    
    # Check if we are in a redirected background environment
    is_redirected = os.environ.get('PYTHONUNBUFFERED') == '1' and os.environ.get('ATLANTICO_DEVICE_LOG')
    
    if want_file:
        # create directory first
        try:
            os.makedirs(os.path.dirname(abs_log_path) or '.', exist_ok=True)
        except Exception:
            pass
            
        try:
            fh = logging.FileHandler(abs_log_path)
            fh.setFormatter(_fmt)
            fh.setLevel(logging.INFO)
            rl.addHandler(fh)
        except Exception:
            pass

    # Only add StreamHandler if we aren't already writing to the same file via FileHandler
    # OR if we aren't in a redirected background run where StreamHandler would cause 
    # double entries in the log file because we also have a FileHandler.
    
    # Actually, the most robust way to avoid duplication in redirected logs is:
    # If we added a FileHandler to X, and stdout is redirected to X, StreamHandler(stdout) 
    # will write to X again.
    
    if not (want_file and is_redirected and os.path.abspath(os.environ.get('ATLANTICO_DEVICE_LOG', '')) == abs_log_path):
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(_fmt)
        sh.setLevel(logging.INFO)
        rl.addHandler(sh)


__all__ = ['setup_logging', 'LOG_PATH']
