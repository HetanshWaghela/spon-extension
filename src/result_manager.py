#!/usr/bin/env python3
"""
=============================================================================
Result Management System for SPON Experiments
=============================================================================

This module provides structured saving, loading, and analysis of experimental
results. Designed for AI research workflows where reproducibility and 
systematic result tracking are critical.

Key Features:
- Hierarchical result organization (experiment → run → configuration)
- Automatic metadata capture (git hash, timestamps, hardware info)
- Multiple export formats (JSON, CSV, LaTeX tables)
- Result aggregation and statistical analysis
- Checkpoint management with versioning

Directory Structure Created:
    results/
    ├── experiment_name/
    │   ├── metadata.json           # Experiment-level metadata
    │   ├── runs/
    │   │   ├── run_20240101_120000/
    │   │   │   ├── config.json     # Run configuration
    │   │   │   ├── results.json    # Detailed results
    │   │   │   ├── metrics.csv     # Time-series metrics
    │   │   │   ├── checkpoints/    # Model checkpoints
    │   │   │   └── figures/        # Generated plots
    │   │   └── ...
    │   ├── aggregated/
    │   │   ├── summary.csv         # All runs summarized
    │   │   ├── pareto.json         # Pareto frontier
    │   │   └── latex_tables/       # Publication-ready tables
    │   └── analysis/
    │       ├── comparisons.json    # Statistical comparisons
    │       └── significance.json   # p-values, confidence intervals

Usage:
    from src.result_manager import ExperimentManager
    
    # Create experiment
    exp = ExperimentManager("allocation_sweep", base_dir="results")
    
    # Start a run
    run = exp.start_run(config={"model": "llama-1b", "sparsity": 0.5})
    
    # Log results
    run.log_metric("perplexity", 15.2, step=0)
    run.log_config_result("TOP-50", {"ppl": 15.2, "params": 0.5})
    
    # Finish and save
    run.finish()
    
    # Aggregate across runs
    exp.aggregate_results()
    exp.export_latex_table("main_results")
"""

import json
import csv
import os
import subprocess
import platform
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Structured Results
# =============================================================================

@dataclass
class HardwareInfo:
    """Captures hardware configuration for reproducibility."""
    platform: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cpu_count: int = 0
    
    @classmethod
    def capture(cls) -> "HardwareInfo":
        """Automatically capture current hardware info."""
        import sys
        info = cls(
            platform=platform.platform(),
            python_version=sys.version.split()[0],
            cpu_count=os.cpu_count() or 0
        )
        
        try:
            import torch
            info.torch_version = torch.__version__
            if torch.cuda.is_available():
                info.cuda_version = torch.version.cuda or ""
                info.gpu_name = torch.cuda.get_device_name(0)
                info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            elif torch.backends.mps.is_available():
                info.gpu_name = "Apple Silicon (MPS)"
        except ImportError:
            pass
        
        return info


@dataclass
class GitInfo:
    """Captures git state for reproducibility."""
    commit_hash: str = ""
    branch: str = ""
    is_dirty: bool = False
    remote_url: str = ""
    
    @classmethod
    def capture(cls) -> "GitInfo":
        """Automatically capture current git state."""
        info = cls()
        try:
            info.commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            info.branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Check if working directory is dirty
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            info.is_dirty = len(status) > 0
            
            info.remote_url = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return info


@dataclass
class RunConfig:
    """Configuration for a single experimental run."""
    # Model settings
    model_name: str = ""
    model_num_layers: int = 0
    model_hidden_dim: int = 0
    
    # SPON settings
    sparsity: float = 0.5
    config_name: str = ""
    layer_mask: List[int] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    
    # Training settings
    epochs: int = 10
    learning_rate: float = 1e-5
    batch_size: int = 8
    block_size: int = 128
    
    # Calibration data
    calibration_dataset: str = "wikitext"
    calibration_samples: int = 1024
    
    # Additional settings
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigResult:
    """Results for a single SPON configuration."""
    config_name: str
    sparsity: float
    
    # Core metrics
    perplexity: float
    loss: float
    num_tokens: int
    
    # SPON-specific
    num_spon_layers: int = 0
    num_spon_params: int = 0
    relative_params: Optional[float] = None
    
    # Training metrics
    training_loss_final: float = 0.0
    training_time_seconds: float = 0.0
    
    # Comparison to baseline
    ppl_vs_dense: float = 0.0  # PPL / dense_PPL
    ppl_vs_teal: float = 0.0   # PPL / teal_PPL
    ppl_improvement: float = 0.0  # (teal_PPL - PPL) / teal_PPL * 100
    
    # Layer-wise metrics (optional)
    layer_sensitivities: Optional[List[float]] = None
    layer_l2_shifts: Optional[Dict[str, float]] = None
    
    # Downstream tasks (optional)
    downstream_tasks: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, handling None values."""
        d = asdict(self)
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class RunSummary:
    """Summary of an entire experimental run."""
    run_id: str
    timestamp: str
    duration_seconds: float
    
    # Metadata
    hardware: HardwareInfo
    git: GitInfo
    config: RunConfig
    
    # Results
    results: List[ConfigResult]
    
    # Aggregated metrics
    best_config: str = ""
    best_perplexity: float = float('inf')
    pareto_configs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "hardware": asdict(self.hardware),
            "git": asdict(self.git),
            "config": asdict(self.config),
            "results": [r.to_dict() for r in self.results],
            "best_config": self.best_config,
            "best_perplexity": self.best_perplexity,
            "pareto_configs": self.pareto_configs
        }


# =============================================================================
# Run Manager - Handles a Single Experimental Run
# =============================================================================

class RunManager:
    """
    Manages a single experimental run.
    
    Handles:
    - Configuration tracking
    - Metric logging (scalar and time-series)
    - Checkpoint saving
    - Result serialization
    """
    
    def __init__(
        self,
        run_dir: Path,
        config: Optional[RunConfig] = None,
        auto_save: bool = True
    ):
        """
        Initialize a run manager.
        
        Args:
            run_dir: Directory for this run
            config: Run configuration
            auto_save: Whether to auto-save on updates
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        
        # Initialize state
        self.run_id = self.run_dir.name
        self.start_time = datetime.now()
        self.config = config or RunConfig()
        self.auto_save = auto_save
        
        # Capture metadata
        self.hardware = HardwareInfo.capture()
        self.git = GitInfo.capture()
        
        # Results storage
        self.config_results: Dict[str, ConfigResult] = {}
        self.metrics_history: Dict[str, List[Dict]] = {}  # metric_name -> [{step, value, timestamp}]
        self.logs: List[Dict] = []
        
        # Status
        self.is_finished = False
        self.end_time: Optional[datetime] = None
        
        logger.info(f"Started run: {self.run_id}")
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        config_name: Optional[str] = None
    ):
        """
        Log a scalar metric.
        
        Args:
            name: Metric name (e.g., "loss", "perplexity")
            value: Metric value
            step: Training step (optional)
            config_name: Associated config (optional)
        """
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        entry = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "config": config_name
        }
        self.metrics_history[name].append(entry)
        
        if self.auto_save:
            self._save_metrics()
    
    def log_config_result(self, result: ConfigResult):
        """
        Log results for a specific SPON configuration.
        
        Args:
            result: ConfigResult object with all metrics
        """
        key = f"{result.config_name}_s{result.sparsity}"
        self.config_results[key] = result
        
        logger.info(
            f"Logged result: {result.config_name} @ {result.sparsity:.0%} sparsity "
            f"-> PPL={result.perplexity:.2f}"
        )
        
        if self.auto_save:
            self._save_results()
    
    def log_message(self, message: str, level: str = "info"):
        """Log a message with timestamp."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        })
    
    def save_checkpoint(
        self,
        biases: Dict[str, Any],
        config_name: str,
        sparsity: float,
        extra_info: Optional[Dict] = None
    ) -> Path:
        """
        Save SPON biases checkpoint.
        
        Args:
            biases: Dictionary of bias tensors
            config_name: Configuration name
            sparsity: Sparsity level
            extra_info: Additional metadata
        
        Returns:
            Path to saved checkpoint
        """
        import torch
        
        checkpoint_name = f"spon_{config_name}_s{sparsity:.2f}.pt"
        checkpoint_path = self.run_dir / "checkpoints" / checkpoint_name
        
        checkpoint = {
            "biases": {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in biases.items()},
            "config_name": config_name,
            "sparsity": sparsity,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "extra": extra_info or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def save_figure(self, fig, name: str, formats: List[str] = ["png", "pdf"]):
        """
        Save a matplotlib figure in multiple formats.
        
        Args:
            fig: Matplotlib figure
            name: Figure name (without extension)
            formats: List of formats to save
        """
        for fmt in formats:
            path = self.run_dir / "figures" / f"{name}.{fmt}"
            fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure: {name}")
    
    def finish(self) -> RunSummary:
        """
        Finish the run and create summary.
        
        Returns:
            RunSummary with all results
        """
        self.end_time = datetime.now()
        self.is_finished = True
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Find best config
        results_list = list(self.config_results.values())
        if results_list:
            best = min(results_list, key=lambda r: r.perplexity)
            best_config = best.config_name
            best_ppl = best.perplexity
        else:
            best_config = ""
            best_ppl = float('inf')
        
        # Compute Pareto frontier
        pareto_configs = self._compute_pareto_configs()
        
        summary = RunSummary(
            run_id=self.run_id,
            timestamp=self.start_time.isoformat(),
            duration_seconds=duration,
            hardware=self.hardware,
            git=self.git,
            config=self.config,
            results=results_list,
            best_config=best_config,
            best_perplexity=best_ppl,
            pareto_configs=pareto_configs
        )
        
        # Save everything
        self._save_all(summary)
        
        logger.info(f"Finished run: {self.run_id} (duration: {duration:.1f}s)")
        return summary
    
    def _compute_pareto_configs(self) -> List[str]:
        """Compute Pareto-optimal configurations."""
        results = [
            r for r in self.config_results.values()
            if r.relative_params is not None
        ]
        if not results:
            return []
        
        # Sort by relative params
        sorted_results = sorted(results, key=lambda r: r.relative_params)
        
        pareto = []
        min_ppl = float('inf')
        
        for r in sorted_results:
            if r.perplexity < min_ppl:
                pareto.append(r.config_name)
                min_ppl = r.perplexity
        
        return pareto
    
    def _save_metrics(self):
        """Save metrics history to CSV."""
        metrics_path = self.run_dir / "metrics.csv"
        
        # Flatten metrics
        rows = []
        for metric_name, history in self.metrics_history.items():
            for entry in history:
                rows.append({
                    "metric": metric_name,
                    "value": entry["value"],
                    "step": entry.get("step"),
                    "config": entry.get("config"),
                    "timestamp": entry["timestamp"]
                })
        
        if rows:
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["metric", "value", "step", "config", "timestamp"])
                writer.writeheader()
                writer.writerows(rows)
    
    def _save_results(self):
        """Save current results to JSON."""
        results_path = self.run_dir / "results.json"
        
        data = {
            "run_id": self.run_id,
            "timestamp": self.start_time.isoformat(),
            "config": asdict(self.config),
            "results": {k: v.to_dict() for k, v in self.config_results.items()}
        }
        
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_all(self, summary: RunSummary):
        """Save all run data."""
        # Save full summary
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        # Save config
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save logs
        logs_path = self.run_dir / "logs.json"
        with open(logs_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        # Final metrics save
        self._save_metrics()
        self._save_results()


# =============================================================================
# Experiment Manager - Handles Multiple Runs
# =============================================================================

class ExperimentManager:
    """
    Manages an entire experiment with multiple runs.
    
    Provides:
    - Run creation and tracking
    - Cross-run aggregation
    - Statistical analysis
    - Export to publication formats
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "results"):
        """
        Initialize experiment manager.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for all results
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        
        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "runs").mkdir(exist_ok=True)
        (self.experiment_dir / "aggregated").mkdir(exist_ok=True)
        (self.experiment_dir / "analysis").mkdir(exist_ok=True)
        
        # Save experiment metadata
        self._save_metadata()
        
        logger.info(f"Initialized experiment: {experiment_name}")
    
    def _save_metadata(self):
        """Save experiment-level metadata."""
        metadata_path = self.experiment_dir / "metadata.json"
        
        metadata = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "hardware": asdict(HardwareInfo.capture()),
            "git": asdict(GitInfo.capture())
        }
        
        # Don't overwrite if exists
        if not metadata_path.exists():
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def start_run(
        self,
        config: Optional[Union[RunConfig, Dict]] = None,
        run_id: Optional[str] = None
    ) -> RunManager:
        """
        Start a new experimental run.
        
        Args:
            config: Run configuration (RunConfig or dict)
            run_id: Custom run ID (default: timestamp-based)
        
        Returns:
            RunManager for this run
        """
        if run_id is None:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        
        run_dir = self.experiment_dir / "runs" / run_id
        
        # Convert dict to RunConfig if needed
        if isinstance(config, dict):
            config = RunConfig(**config)
        
        return RunManager(run_dir, config)
    
    def list_runs(self) -> List[str]:
        """List all run IDs in this experiment."""
        runs_dir = self.experiment_dir / "runs"
        return [d.name for d in runs_dir.iterdir() if d.is_dir()]
    
    def load_run(self, run_id: str) -> Optional[RunSummary]:
        """Load a completed run's summary."""
        summary_path = self.experiment_dir / "runs" / run_id / "summary.json"
        
        if not summary_path.exists():
            logger.warning(f"Run not found or incomplete: {run_id}")
            return None
        
        with open(summary_path) as f:
            data = json.load(f)
        
        # Reconstruct RunSummary (simplified)
        return data  # Return raw dict for flexibility
    
    def aggregate_results(self) -> Dict:
        """
        Aggregate results across all runs.
        
        Creates:
        - Summary CSV with all configurations
        - Pareto frontier JSON
        - Statistical summaries
        
        Returns:
            Aggregated statistics
        """
        all_results = []
        
        for run_id in self.list_runs():
            run_data = self.load_run(run_id)
            if run_data and "results" in run_data:
                for result in run_data["results"]:
                    result["run_id"] = run_id
                    all_results.append(result)
        
        if not all_results:
            logger.warning("No results to aggregate")
            return {}
        
        # Save summary CSV
        summary_path = self.experiment_dir / "aggregated" / "summary.csv"
        if all_results:
            # Get all keys
            all_keys = set()
            for r in all_results:
                all_keys.update(r.keys())
            
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(all_results)
        
        # Compute aggregate statistics
        stats = self._compute_statistics(all_results)
        
        stats_path = self.experiment_dir / "analysis" / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Compute Pareto frontier
        pareto = self._compute_pareto_frontier(all_results)
        pareto_path = self.experiment_dir / "aggregated" / "pareto.json"
        with open(pareto_path, 'w') as f:
            json.dump(pareto, f, indent=2)
        
        logger.info(f"Aggregated {len(all_results)} results from {len(self.list_runs())} runs")
        return stats
    
    def _compute_statistics(self, results: List[Dict]) -> Dict:
        """Compute statistical summaries."""
        from collections import defaultdict
        
        # Group by config
        by_config = defaultdict(list)
        for r in results:
            key = f"{r.get('config_name', 'unknown')}_s{r.get('sparsity', 0)}"
            by_config[key].append(r)
        
        stats = {}
        for config_key, config_results in by_config.items():
            ppls = [r.get('perplexity') for r in config_results if r.get('perplexity') is not None]
            
            if ppls:
                stats[config_key] = {
                    "n_runs": len(ppls),
                    "ppl_mean": float(np.mean(ppls)),
                    "ppl_std": float(np.std(ppls)),
                    "ppl_min": float(np.min(ppls)),
                    "ppl_max": float(np.max(ppls)),
                    "ppl_median": float(np.median(ppls))
                }
        
        return stats
    
    def _compute_pareto_frontier(self, results: List[Dict]) -> List[Dict]:
        """Compute Pareto-optimal configurations."""
        # Filter valid results
        valid = [
            r for r in results
            if r.get('perplexity') is not None and r.get('relative_params') is not None
        ]
        
        if not valid:
            return []
        
        # Sort by params
        sorted_results = sorted(valid, key=lambda r: r['relative_params'])
        
        pareto = []
        min_ppl = float('inf')
        
        for r in sorted_results:
            if r['perplexity'] < min_ppl:
                pareto.append({
                    "config_name": r.get('config_name'),
                    "sparsity": r.get('sparsity'),
                    "perplexity": r['perplexity'],
                    "relative_params": r['relative_params']
                })
                min_ppl = r['perplexity']
        
        return pareto
    
    def export_latex_table(
        self,
        table_name: str = "main_results",
        metric_keys: List[str] = ["perplexity", "relative_params"],
        caption: str = "SPON Allocation Results"
    ) -> str:
        """
        Export results as LaTeX table.
        
        Args:
            table_name: Output file name
            metric_keys: Metrics to include
            caption: Table caption
        
        Returns:
            LaTeX table string
        """
        # Load aggregated data
        summary_path = self.experiment_dir / "aggregated" / "summary.csv"
        
        if not summary_path.exists():
            self.aggregate_results()
        
        rows = []
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return ""
        
        # Build LaTeX table
        cols = ["config_name", "sparsity"] + metric_keys
        header = " & ".join([c.replace("_", " ").title() for c in cols])
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\label{tab:" + table_name + "}",
            "\\begin{tabular}{" + "l" * len(cols) + "}",
            "\\toprule",
            header + " \\\\",
            "\\midrule"
        ]
        
        for row in rows:
            values = []
            for col in cols:
                val = row.get(col, "-")
                # Format numbers
                try:
                    val = float(val)
                    if col == "perplexity":
                        val = f"{val:.2f}"
                    elif col == "relative_params":
                        val = f"{val:.2f}"
                    else:
                        val = f"{val:.3f}"
                except (ValueError, TypeError):
                    pass
                values.append(str(val))
            latex_lines.append(" & ".join(values) + " \\\\")
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        latex_str = "\n".join(latex_lines)
        
        # Save
        latex_dir = self.experiment_dir / "aggregated" / "latex_tables"
        latex_dir.mkdir(exist_ok=True)
        
        output_path = latex_dir / f"{table_name}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        logger.info(f"Exported LaTeX table: {output_path}")
        return latex_str
    
    def compare_configs(
        self,
        baseline: str = "UNIF-ALL",
        test_configs: Optional[List[str]] = None
    ) -> Dict:
        """
        Statistically compare configurations against baseline.
        
        Args:
            baseline: Baseline configuration name
            test_configs: Configurations to compare (default: all)
        
        Returns:
            Comparison results with effect sizes
        """
        from scipy import stats as scipy_stats
        
        # Load all results
        all_results = []
        for run_id in self.list_runs():
            run_data = self.load_run(run_id)
            if run_data and "results" in run_data:
                all_results.extend(run_data["results"])
        
        # Group by config
        from collections import defaultdict
        by_config = defaultdict(list)
        for r in all_results:
            key = r.get('config_name', 'unknown')
            if r.get('perplexity'):
                by_config[key].append(r['perplexity'])
        
        if baseline not in by_config:
            logger.warning(f"Baseline {baseline} not found")
            return {}
        
        baseline_ppls = by_config[baseline]
        comparisons = {}
        
        configs_to_test = test_configs or [k for k in by_config.keys() if k != baseline]
        
        for config in configs_to_test:
            if config not in by_config:
                continue
            
            test_ppls = by_config[config]
            
            # Basic statistics
            comp = {
                "baseline_mean": float(np.mean(baseline_ppls)),
                "baseline_std": float(np.std(baseline_ppls)),
                "test_mean": float(np.mean(test_ppls)),
                "test_std": float(np.std(test_ppls)),
                "n_baseline": len(baseline_ppls),
                "n_test": len(test_ppls)
            }
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(baseline_ppls) - 1) * np.var(baseline_ppls) + 
                 (len(test_ppls) - 1) * np.var(test_ppls)) /
                (len(baseline_ppls) + len(test_ppls) - 2)
            )
            if pooled_std > 0:
                comp["cohens_d"] = (np.mean(test_ppls) - np.mean(baseline_ppls)) / pooled_std
            
            # t-test (if enough samples)
            if len(baseline_ppls) >= 2 and len(test_ppls) >= 2:
                t_stat, p_value = scipy_stats.ttest_ind(baseline_ppls, test_ppls)
                comp["t_statistic"] = float(t_stat)
                comp["p_value"] = float(p_value)
                comp["significant_0.05"] = p_value < 0.05
            
            comparisons[config] = comp
        
        # Save
        comp_path = self.experiment_dir / "analysis" / "comparisons.json"
        with open(comp_path, 'w') as f:
            json.dump(comparisons, f, indent=2)
        
        return comparisons


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_save_results(
    results: List[Dict],
    output_dir: str,
    experiment_name: str = "experiment"
) -> Path:
    """
    Quick helper to save results without full experiment setup.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory
        experiment_name: Experiment name
    
    Returns:
        Path to saved results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_path / f"{experiment_name}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "experiment": experiment_name,
            "timestamp": timestamp,
            "hardware": asdict(HardwareInfo.capture()),
            "git": asdict(GitInfo.capture()),
            "results": results
        }, f, indent=2, default=str)
    
    # Save CSV
    csv_path = output_path / f"{experiment_name}_{timestamp}.csv"
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(results)
    
    logger.info(f"Saved results to {json_path}")
    return json_path
