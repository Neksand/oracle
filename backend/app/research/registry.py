#!/usr/bin/env python3
"""
Model Registry for ORACLE volatility forecasting models.

JSONL-based registry with versioning, promotion, and rollback capabilities.
Integrates with MLflow-compatible logging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = REGISTRY_DIR / "models.jsonl"
CHECKPOINTS_DIR = REGISTRY_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    """Single model registry entry."""
    name: str
    version: str
    git_sha: str
    stage: str  # "development", "staging", "production", "archived"
    hyperparameters: dict
    metrics: dict
    checkpoint_path: str
    scaler_path: str | None
    created_at: str
    promoted_at: str | None
    promoted_by: str | None
    description: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_jsonl(line: str) -> "ModelEntry":
        data = json.loads(line)
        return ModelEntry(**data)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    """Check if working directory has uncommitted changes."""
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
    except Exception:
        return False


def compute_file_hash(path: Path) -> str:
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def generate_version(name: str, metrics: dict) -> str:
    """Generate semantic version from metrics."""
    # Format: v{major}.{minor}.{patch}-{metric_hash}
    # Major: architecture changes, Minor: hyperparam changes, Patch: retrain
    rmse = metrics.get("rmse", 0)
    qlike = metrics.get("qlike", 0)
    metric_hash = hashlib.md5(f"{rmse:.6f}{qlike:.6f}".encode()).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"v1.0.{timestamp}-{metric_hash}"


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Model registry with CRUD operations."""

    def __init__(self, registry_file: Path = REGISTRY_FILE):
        self.registry_file = registry_file
        self._cache: list[ModelEntry] | None = None

    def _load(self) -> list[ModelEntry]:
        if self._cache is not None:
            return self._cache

        entries = []
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(ModelEntry.from_jsonl(line))
                        except Exception as e:
                            print(f"Warning: failed to parse registry line: {e}", file=sys.stderr)
        self._cache = entries
        return entries

    def _save(self, entries: list[ModelEntry]) -> None:
        with open(self.registry_file, "w") as f:
            for entry in entries:
                f.write(entry.to_jsonl() + "\n")
        self._cache = entries

    def register(
        self,
        name: str,
        hyperparameters: dict,
        metrics: dict,
        checkpoint_path: str,
        scaler_path: str | None = None,
        description: str = "",
        stage: str = "development",
    ) -> ModelEntry:
        """Register a new model version."""
        git_sha = get_git_sha()
        version = generate_version(name, metrics)
        created_at = datetime.now(timezone.utc).isoformat()

        # Copy checkpoint to registry
        src_ckpt = Path(checkpoint_path)
        if not src_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        dst_ckpt = CHECKPOINTS_DIR / f"{name}_{version}{src_ckpt.suffix}"
        shutil.copy2(src_ckpt, dst_ckpt)

        scaler_dst = None
        if scaler_path:
            src_scaler = Path(scaler_path)
            if src_scaler.exists():
                scaler_dst = CHECKPOINTS_DIR / f"{name}_{version}_scaler{src_scaler.suffix}"
                shutil.copy2(src_scaler, scaler_dst)

        entry = ModelEntry(
            name=name,
            version=version,
            git_sha=git_sha,
            stage=stage,
            hyperparameters=hyperparameters,
            metrics=metrics,
            checkpoint_path=str(dst_ckpt.relative_to(REGISTRY_DIR)),
            scaler_path=str(scaler_dst.relative_to(REGISTRY_DIR)) if scaler_dst else None,
            created_at=created_at,
            promoted_at=None,
            promoted_by=None,
            description=description,
        )

        entries = self._load()
        entries.append(entry)
        self._save(entries)

        print(f"Registered {name} {version} ({stage})")
        print(f"  Checkpoint: {dst_ckpt}")
        print(f"  RMSE: {metrics.get('rmse', 'N/A')}, QLIKE: {metrics.get('qlike', 'N/A')}")
        return entry

    def list(self, name: str | None = None, stage: str | None = None) -> list[ModelEntry]:
        """List models with optional filters."""
        entries = self._load()
        if name:
            entries = [e for e in entries if e.name == name]
        if stage:
            entries = [e for e in entries if e.stage == stage]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def get(self, name: str, version: str) -> ModelEntry | None:
        """Get specific model version."""
        for entry in self._load():
            if entry.name == name and entry.version == version:
                return entry
        return None

    def get_latest(self, name: str, stage: str = "production") -> ModelEntry | None:
        """Get latest model in a stage."""
        entries = self.list(name=name, stage=stage)
        return entries[0] if entries else None

    def promote(self, name: str, version: str, target_stage: str, promoted_by: str = "cli") -> ModelEntry | None:
        """Promote a model version to a new stage."""
        valid_stages = ["development", "staging", "production", "archived"]
        if target_stage not in valid_stages:
            raise ValueError(f"Invalid stage: {target_stage}. Must be one of {valid_stages}")

        entries = self._load()
        for entry in entries:
            if entry.name == name and entry.version == version:
                # Demote current model in target stage (if production)
                if target_stage == "production":
                    for other in entries:
                        if other.name == name and other.stage == "production" and other.version != version:
                            other.stage = "staging"
                            other.promoted_at = datetime.now(timezone.utc).isoformat()
                            other.promoted_by = promoted_by

                entry.stage = target_stage
                entry.promoted_at = datetime.now(timezone.utc).isoformat()
                entry.promoted_by = promoted_by
                self._save(entries)
                print(f"Promoted {name} {version} -> {target_stage}")
                return entry

        print(f"Model {name} {version} not found")
        return None

    def rollback(self, name: str, target_version: str) -> ModelEntry | None:
        """Rollback production to a previous version."""
        return self.promote(name, target_version, "production", "rollback")

    def archive(self, name: str, version: str) -> ModelEntry | None:
        """Archive a model version."""
        return self.promote(name, version, "archived")

    def delete(self, name: str, version: str, force: bool = False) -> bool:
        """Delete a model version (only if not in production)."""
        entries = self._load()
        for i, entry in enumerate(entries):
            if entry.name == name and entry.version == version:
                if entry.stage == "production" and not force:
                    print(f"Cannot delete production model {name} {version} without --force")
                    return False

                # Delete checkpoint files
                ckpt_path = REGISTRY_DIR / entry.checkpoint_path
                if ckpt_path.exists():
                    ckpt_path.unlink()

                if entry.scaler_path:
                    scaler_path = REGISTRY_DIR / entry.scaler_path
                    if scaler_path.exists():
                        scaler_path.unlink()

                entries.pop(i)
                self._save(entries)
                print(f"Deleted {name} {version}")
                return True

        print(f"Model {name} {version} not found")
        return False

    def export_metadata(self, output_path: Path) -> None:
        """Export registry metadata as JSON."""
        entries = self._load()
        data = [asdict(e) for e in entries]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(entries)} entries to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_register(args: argparse.Namespace) -> int:
    registry = ModelRegistry()

    # Load metrics from file if provided
    metrics = {}
    if args.metrics_file:
        with open(args.metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = json.loads(args.metrics) if args.metrics else {}

    hyperparams = json.loads(args.hyperparams) if args.hyperparams else {}

    try:
        registry.register(
            name=args.name,
            hyperparameters=hyperparams,
            metrics=metrics,
            checkpoint_path=args.checkpoint,
            scaler_path=args.scaler,
            description=args.description,
            stage=args.stage,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    entries = registry.list(name=args.name, stage=args.stage)

    if not entries:
        print("No models found")
        return 0

    # Print table
    print(f"{'Name':<20} {'Version':<25} {'Stage':<12} {'RMSE':<10} {'QLIKE':<10} {'Created':<20} {'Description'}")
    print("-" * 120)
    for e in entries:
        rmse = f"{e.metrics.get('rmse', 0):.4f}" if 'rmse' in e.metrics else "N/A"
        qlike = f"{e.metrics.get('qlike', 0):.4f}" if 'qlike' in e.metrics else "N/A"
        created = e.created_at[:19].replace("T", " ")
        desc = e.description[:40] if e.description else ""
        print(f"{e.name:<20} {e.version:<25} {e.stage:<12} {rmse:<10} {qlike:<10} {created:<20} {desc}")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    entry = registry.get(args.name, args.version)
    if not entry:
        print(f"Model {args.name} {args.version} not found")
        return 1

    print(json.dumps(asdict(entry), indent=2))
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    result = registry.promote(args.name, args.version, args.stage, args.promoted_by)
    return 0 if result else 1


def cmd_rollback(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    result = registry.rollback(args.name, args.version)
    return 0 if result else 1


def cmd_archive(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    result = registry.archive(args.name, args.version)
    return 0 if result else 1


def cmd_delete(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    result = registry.delete(args.name, args.version, args.force)
    return 0 if result else 1


def cmd_export(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    registry.export_metadata(Path(args.output))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.research.registry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register
    p_reg = subparsers.add_parser("register", help="Register a new model version")
    p_reg.add_argument("--name", required=True, help="Model name (e.g., lstm-x, volformer)")
    p_reg.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    p_reg.add_argument("--scaler", help="Path to scaler pickle file")
    p_reg.add_argument("--metrics", help="JSON string of metrics")
    p_reg.add_argument("--metrics-file", help="Path to metrics JSON file")
    p_reg.add_argument("--hyperparams", help="JSON string of hyperparameters")
    p_reg.add_argument("--description", default="", help="Model description")
    p_reg.add_argument("--stage", default="development", choices=["development", "staging", "production", "archived"])
    p_reg.set_defaults(func=cmd_register)

    # list
    p_list = subparsers.add_parser("list", help="List registered models")
    p_list.add_argument("--name", help="Filter by model name")
    p_list.add_argument("--stage", help="Filter by stage")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = subparsers.add_parser("show", help="Show model details")
    p_show.add_argument("--name", required=True)
    p_show.add_argument("--version", required=True)
    p_show.set_defaults(func=cmd_show)

    # promote
    p_prom = subparsers.add_parser("promote", help="Promote model to stage")
    p_prom.add_argument("--name", required=True)
    p_prom.add_argument("--version", required=True)
    p_prom.add_argument("--stage", required=True, choices=["development", "staging", "production", "archived"])
    p_prom.add_argument("--promoted-by", default="cli")
    p_prom.set_defaults(func=cmd_promote)

    # rollback
    p_roll = subparsers.add_parser("rollback", help="Rollback production to version")
    p_roll.add_argument("--name", required=True)
    p_roll.add_argument("--version", required=True)
    p_roll.set_defaults(func=cmd_rollback)

    # archive
    p_arch = subparsers.add_parser("archive", help="Archive a model version")
    p_arch.add_argument("--name", required=True)
    p_arch.add_argument("--version", required=True)
    p_arch.set_defaults(func=cmd_archive)

    # delete
    p_del = subparsers.add_parser("delete", help="Delete a model version")
    p_del.add_argument("--name", required=True)
    p_del.add_argument("--version", required=True)
    p_del.add_argument("--force", action="store_true", help="Force delete production model")
    p_del.set_defaults(func=cmd_delete)

    # export
    p_exp = subparsers.add_parser("export", help="Export registry to JSON")
    p_exp.add_argument("--output", required=True, help="Output JSON file path")
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())