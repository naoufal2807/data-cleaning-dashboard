from __future__ import annotations
import argparse, yaml
from pathlib import Path
from src.cleaner import run_cleaning

def parse_args():
    p = argparse.ArgumentParser(description="Data Cleaning & Automation — YAML-driven")
    p.add_argument("--input", required=True, help="Path to input CSV file")
    p.add_argument("--outdir", default="outputs", help="Directory to write outputs")
    p.add_argument("--config", default="data_cleaning_config.yaml", help="Path to YAML config")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    run_cleaning(Path(args.input), Path(args.outdir), config=cfg)

if __name__ == "__main__":
    main()
