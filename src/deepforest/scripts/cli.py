import argparse
import os

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from deepforest.conf.schema import Config as StructuredConfig
from deepforest.main import deepforest

# plot_results is provided by the (legacy) top-level module `visualize.py`.
# We import it lazily inside `predict` to avoid import-time dependency on the
# legacy loader (which is opt-in). See predict() for the runtime import logic.


def train(config: DictConfig) -> None:
    """Train a DeepForest model with the given configuration.

    Args:
        config: Hydra configuration object containing training parameters
    """
    m = deepforest(config=config)
    m.trainer.fit(m)


def predict(
    config: DictConfig,
    input_path: str,
    output_path: str | None = None,
    plot: bool | None = False,
) -> None:
    """Run prediction for the given image, optionally saving the results to the
    provided path and optionally visualizing the results.

    Args:
        config (DictConfig): Hydra configuration.
        input_path (str): Path to the input image.
        output_path (Optional[str]): Path to save the prediction results.
        plot (Optional[bool]): Whether to plot the results.

    Returns:
        None
    """
    m = deepforest(config=config)
    res = m.predict_tile(
        path=input_path,
        patch_size=config.patch_size,
        patch_overlap=config.patch_overlap,
        iou_threshold=config.nms_thresh,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res.to_csv(output_path, index=False)

    if plot:
        # Lazy-import plot_results. Try package export first; if unavailable
        # (legacy loader not enabled), attempt to load the old top-level
        # `visualize.py` module and fetch plot_results from it.
        try:
            from deepforest.visualize import plot_results  # type: ignore
        except Exception:
            # Try to load legacy visualize.py directly from package source
            import importlib.util

            _here = os.path.dirname(__file__)
            _viz_path = os.path.normpath(os.path.join(_here, "..", "visualize.py"))
            plot_results = None
            if os.path.exists(_viz_path):
                try:
                    spec = importlib.util.spec_from_file_location(
                        "deepforest._visualize_legacy", _viz_path
                    )
                    viz = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(viz)  # type: ignore[attr-defined]
                    plot_results = getattr(viz, "plot_results", None)
                except Exception:
                    plot_results = None

        if plot_results is None:
            raise RuntimeError(
                "plot_results is not available. Either enable the legacy loader by setting DEEPFOREST_LOAD_LEGACY_VISUALIZE=1 or ensure visualize exports plot_results."
            )
        plot_results(res)


def main():
    """Main CLI entry point for DeepForest.

    Provides subcommands for training, prediction, and configuration
    management.
    """
    parser = argparse.ArgumentParser(description="DeepForest CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    _ = subparsers.add_parser(
        "train",
        help="Train a model",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run prediction on input",
        epilog="Any remaining arguments <key>=<value> will be passed to Hydra to override the current config.",
    )
    predict_parser.add_argument("input", help="Path to input raster")
    predict_parser.add_argument("-o", "--output", help="Path to prediction results")
    predict_parser.add_argument("--plot", action="store_true", help="Plot results")

    # Show config subcommand
    subparsers.add_parser("config", help="Show the current config")

    # Gallery subcommands
    gallery_parser = subparsers.add_parser("gallery", help="Gallery utilities")
    gallery_sub = gallery_parser.add_subparsers(dest="gallery_cmd")

    gallery_export = gallery_sub.add_parser(
        "export", help="Export predictions to a local gallery (thumbnails + metadata)"
    )
    gallery_export.add_argument(
        "-i",
        "--input",
        help="Path to predictions CSV/JSON (rows with image_path and bbox)",
    )
    gallery_export.add_argument(
        "-o", "--out", dest="out", help="Output directory for gallery", required=True
    )
    gallery_export.add_argument(
        "--root-dir",
        dest="root_dir",
        help="Root directory to resolve relative image paths",
    )
    gallery_export.add_argument(
        "--max-crops", type=int, default=None, help="Maximum number of crops to export"
    )
    gallery_export.add_argument(
        "--sample-by-image",
        action="store_true",
        help="Sample by image to distribute crops across images",
    )
    gallery_export.add_argument(
        "--per-image-limit",
        type=int,
        default=None,
        help="Limit crops per image when sampling by image",
    )
    gallery_export.add_argument(
        "--sample-seed", type=int, default=None, help="Seed for deterministic sampling"
    )
    gallery_export.add_argument(
        "--start-server",
        action="store_true",
        help="Start a tiny local HTTP server to view the gallery",
    )
    gallery_export.add_argument(
        "--port", type=int, default=0, help="Port to serve the gallery on (0 = auto)"
    )
    gallery_export.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser when starting server",
    )

    # Config options for Hydra
    parser.add_argument("--config-dir", help="Show available config overrides and exit")
    parser.add_argument(
        "--config-name", help="Show available config overrides and exit", default="config"
    )

    args, overrides = parser.parse_known_args()

    if args.config_dir is not None:
        initialize_config_dir(version_base=None, config_dir=args.config_dir)
    else:
        initialize(version_base=None, config_path="pkg://deepforest.conf")

    base = OmegaConf.structured(StructuredConfig)
    cfg = compose(config_name=args.config_name, overrides=overrides)
    cfg = OmegaConf.merge(base, cfg)

    if args.command == "predict":
        predict(cfg, input_path=args.input, output_path=args.output, plot=args.plot)
    elif args.command == "train":
        train(cfg)
    elif args.command == "config":
        print(OmegaConf.to_yaml(cfg, resolve=True))
    elif args.command == "gallery":
        # Gallery subcommands
        if args.gallery_cmd == "export":
            try:
                import pandas as pd
            except Exception as exc:
                raise RuntimeError(
                    "pandas is required for gallery export. Please install it in your environment."
                ) from exc

            if args.input is None:
                raise RuntimeError(
                    "Please provide an input predictions file with -i/--input"
                )

            # read CSV or JSON depending on extension
            input_path = args.input
            if input_path.lower().endswith(".json") or input_path.lower().endswith(
                ".jsonl"
            ):
                df = pd.read_json(input_path, lines=input_path.lower().endswith(".jsonl"))
            else:
                df = pd.read_csv(input_path)

            from deepforest.visualize import (
                export_to_gallery,
                start_local_gallery,
                write_gallery_html,
            )

            outdir = args.out
            export_to_gallery(
                df,
                outdir,
                root_dir=args.root_dir,
                max_crops=args.max_crops,
                sample_seed=args.sample_seed,
                sample_by_image=args.sample_by_image,
                per_image_limit=args.per_image_limit,
            )
            write_gallery_html(outdir)

            if args.start_server:
                start_local_gallery(
                    outdir, port=args.port, open_browser=not args.no_browser
                )


if __name__ == "__main__":
    main()
