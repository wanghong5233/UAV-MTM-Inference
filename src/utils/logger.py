"""
Logging utilities.

Supports TensorBoard / Weights & Biases backends, plus
CSV metric export and structured debug event logging.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from numbers import Number
from typing import Any, Dict, Optional
from pathlib import Path


class Logger:
    """Unified logger wrapper."""
    
    def __init__(self, config: Dict, log_dir: Optional[str] = None):
        """Initialize logger backend."""
        self.config = config
        logging_config = config.get('logging', {})
        self.writer = None
        self.wandb = None
        
        # Backend
        self.backend = logging_config.get('backend', 'tensorboard')
        self.log_interval = int(logging_config.get('log_interval', 10))

        # File outputs
        self.log_dir = Path(log_dir or logging_config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_enabled = bool(logging_config.get("csv_enabled", True))
        self.csv_path = self.log_dir / str(logging_config.get("csv_filename", "metrics_long.csv"))
        self.debug_events_enabled = bool(logging_config.get("debug_events_enabled", True))
        self.debug_events_path = self.log_dir / str(
            logging_config.get("debug_events_filename", "debug_events.jsonl")
        )

        if self.csv_enabled and not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "step", "key", "value"])
        if self.debug_events_enabled and not self.debug_events_path.exists():
            self.debug_events_path.touch()
        
        # Initialize backend
        if self.backend == 'tensorboard':
            self._init_tensorboard(log_dir)
        elif self.backend == 'wandb':
            self._init_wandb(logging_config)
        else:
            print(f"[WARN] Unknown logging backend: {self.backend}")
    
    def _init_tensorboard(self, log_dir: Optional[str]):
        """Initialize TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[INFO] Tensorboard initialized at {log_dir}")
        except ImportError:
            print("[WARN] tensorboard not installed")
            self.writer = None
    
    def _init_wandb(self, logging_config: Dict):
        """Initialize Weights & Biases."""
        try:
            import wandb
            wandb.init(
                project=logging_config.get('project', 'UAV-MTL-Inference'),
                entity=logging_config.get('entity'),
                config=self.config,
            )
            self.wandb = wandb
            print("[INFO] WandB initialized")
        except ImportError:
            print("[WARN] wandb not installed")
            self.wandb = None

    def _coerce_scalar(self, value: Any) -> Optional[float]:
        """Convert scalar-like values to float; return None if not scalar."""
        if isinstance(value, Number):
            return float(value)
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        return None

    def _write_csv_rows(self, metrics: Dict[str, float], step: int) -> None:
        """Append metric rows to long-format CSV.

        Force flush + fsync after each iteration so that a SIGKILL / OOM /
        node reboot loses at most the rows from the in-flight update, never
        a 30-second OS buffer. PPO iterations are 10-60s apart, so the
        fsync overhead is negligible (<1 ms per call).
        """
        if not self.csv_enabled:
            return
        ts = datetime.now().isoformat()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for key, val in metrics.items():
                writer.writerow([ts, int(step), key, float(val)])
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    
    def log_metrics(self, metrics: Dict, step: int, prefix: str = ''):
        """Log scalar metrics."""
        if not metrics:
            return

        scalar_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            scalar = self._coerce_scalar(value)
            if scalar is None:
                continue
            k = f"{prefix}{key}" if prefix else key
            scalar_metrics[k] = scalar

        if not scalar_metrics:
            return

        if self.backend == 'tensorboard' and self.writer is not None:
            for key, value in scalar_metrics.items():
                self.writer.add_scalar(key, value, step)
            self.writer.flush()

        elif self.backend == 'wandb' and self.wandb is not None:
            log_dict = dict(scalar_metrics)
            log_dict['step'] = int(step)
            self.wandb.log(log_dict)

        self._write_csv_rows(scalar_metrics, step=step)

    def log_event(
        self,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        level: str = "INFO",
    ) -> None:
        """Write structured debug event records to JSONL."""
        if not self.debug_events_enabled:
            return
        record = {
            "timestamp": datetime.now().isoformat(),
            "level": str(level),
            "event": str(event),
            "step": None if step is None else int(step),
            "payload": payload or {},
        }
        with open(self.debug_events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    
    def close(self):
        """Close logger resources."""
        if self.backend == 'tensorboard' and self.writer is not None:
            self.writer.close()
        elif self.backend == 'wandb' and self.wandb is not None:
            self.wandb.finish()

