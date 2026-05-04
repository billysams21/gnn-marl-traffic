"""
Logging and metric tracking utilities.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional


class MetricLogger:
    """Simple CSV + JSON logger for training metrics."""

    def __init__(self, log_dir: str, experiment_name: Optional[str] = "experiment"):
        self.log_dir = os.path.join(log_dir, experiment_name) if experiment_name else log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.config_path = os.path.join(self.log_dir, "config.json")

        self._csv_initialized = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        self._episode_metrics: List[Dict] = []

    def save_config(self, config: dict, overwrite: bool = True):
        """Save experiment configuration."""
        if not overwrite and os.path.exists(self.config_path):
            return
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """Log metrics for one episode."""
        metrics["episode"] = episode
        metrics["timestamp"] = datetime.now().isoformat()

        self._episode_metrics.append(metrics)

        # Write/append to CSV
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)
            self._csv_initialized = True
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writerow(metrics)

        # Print summary
        summary = f"Episode {episode:4d}"
        for k, v in metrics.items():
            if k not in ("episode", "timestamp"):
                if isinstance(v, float):
                    summary += f" | {k}: {v:.4f}"
                else:
                    summary += f" | {k}: {v}"
        print(summary)

    def get_metrics(self) -> List[Dict]:
        return self._episode_metrics
