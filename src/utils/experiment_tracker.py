"""
Experiment tracking utilities.

Store config snapshots, run metadata, and summary results.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ExperimentTracker:
    """Experiment tracker with lightweight metadata persistence."""
    
    def __init__(self, experiment_name: str, config: Dict, output_dir: str):
        """Initialize experiment tracker."""
        self.experiment_name = experiment_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration': None,
        }

        # Persist config snapshot
        self._save_config()
    
    def _save_config(self):
        """Save config snapshot."""
        config_path = self.output_dir / 'config.yaml'
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"[INFO] Config snapshot saved to {config_path}")
    
    def log_result(self, results: Dict[str, Any]):
        """Save summarized run results."""
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def finish(self):
        """Finalize tracker and write metadata."""
        self.metadata['end_time'] = datetime.now().isoformat()

        # Compute run duration
        start = datetime.fromisoformat(self.metadata['start_time'])
        end = datetime.fromisoformat(self.metadata['end_time'])
        self.metadata['duration'] = (end - start).total_seconds()

        # Persist metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"[INFO] Experiment finished, duration={self.metadata['duration']:.2f}s")

