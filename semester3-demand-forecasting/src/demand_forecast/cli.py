"""Command-line entry point for the Semester 3 forecasting study."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import zipfile

from demand_forecast.config import apply_demo_overrides, load_config, resolve_config_arg
from demand_forecast.data.m5 import download_instructions
from demand_forecast.paths import project_root, resolve_path
from demand_forecast.study import run_eda_stage, run_forecast_study, run_prepare


def _download_m5(config: dict) -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(download_instructions())
    raw = resolve_path(project_root(), config["data"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "m5-forecasting-accuracy", "-p", str(raw)],
        check=True,
    )
    zip_path = raw / "m5-forecasting-accuracy.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw)
    else:
        for path in raw.glob("*.zip"):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(raw)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Semester 3 retail demand-forecasting study.")
    parser.add_argument("--config", default="config/semester3.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download M5 with the Kaggle CLI, if it is installed.")
    sub.add_parser("prepare", help="Build the study panel from M5 or synthetic data.")
    sub.add_parser("eda", help="Write exploratory figures for the prepared panel.")
    forecast = sub.add_parser("forecast", help="Run the chronological model comparison.")
    forecast.add_argument("--skip-chronos", action="store_true", help="Run baselines only.")
    forecast.add_argument("--rebuild", action="store_true", help="Rebuild the panel before forecasting.")
    sub.add_parser("all", help="Prepare, explore, and forecast.")
    demo = sub.add_parser("demo", help="Run the full pipeline on small synthetic data.")
    demo.add_argument("--with-chronos", action="store_true", help="Also run Chronos-2 on the synthetic demo.")

    args = parser.parse_args(argv)
    config = load_config(resolve_config_arg(args.config))
    if args.command == "demo":
        config = apply_demo_overrides(config)
        if args.with_chronos:
            config["models"]["chronos"] = True
        run_prepare(config)
        run_eda_stage(config)
        run_forecast_study(config)
        return
    if args.command == "download":
        _download_m5(config)
        return
    if args.command == "prepare":
        run_prepare(config)
        return
    if args.command == "eda":
        run_eda_stage(config)
        return
    if args.command == "forecast":
        if args.skip_chronos:
            config["models"]["chronos"] = False
        run_forecast_study(config, rebuild=args.rebuild)
        return
    if args.command == "all":
        run_prepare(config)
        run_eda_stage(config)
        run_forecast_study(config)
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
