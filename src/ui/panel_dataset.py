"""
Panel 1: Dataset Management
- Dataset discovery and validation
- Train/test/validation splitting
- .npy file extraction
- Dataset health briefing
"""
import gradio as gr
from pathlib import Path
from typing import Tuple
import sys
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.splitter import DatasetScanner, DatasetSplitter
from src.data.health_checker import DatasetHealthChecker
from src.data.npy_extractor import NpyExtractor
import logging

logger = logging.getLogger(__name__)

# Global state for panel
_current_scanner = None
_current_dataset_info = None


def create_dataset_panel() -> gr.Blocks:
    """
    Create Panel 1: Dataset Management

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 📊 Dataset Management")
        gr.Markdown("Manage your wakeword datasets, split them, and validate quality.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Dataset Selection")
                dataset_root = gr.Textbox(
                    label="Dataset Root Directory",
                    placeholder="C:/path/to/datasets or data/raw",
                    lines=1,
                    value="data/raw"  # Default value
                )

                gr.Markdown("**Expected Structure:**")
                gr.Markdown("""
                ```
                dataset_root/
                ├── positive/       (wakeword utterances)
                │   └── subfolders...
                ├── negative/       (non-wakeword speech)
                │   └── subfolders...
                ├── hard_negative/  (similar sounding phrases)
                │   └── subfolders...
                ├── background/     (environmental noise)
                │   └── subfolders...
                ├── rirs/           (room impulse responses)
                │   └── subfolders...
                └── npy/            (.npy feature files, optional)
                ```
                **Note:** Subfolders are automatically scanned recursively!
                """)

                skip_validation = gr.Checkbox(
                    label="Fast Scan (Skip Validation)",
                    value=False,
                    info="Only count files without validation (much faster for large datasets)"
                )

                scan_button = gr.Button("🔍 Scan Datasets", variant="primary")
                scan_status = gr.Textbox(
                    label="Scan Status",
                    value="Ready to scan",
                    lines=1,
                    interactive=False
                )

            with gr.Column():
                gr.Markdown("### Dataset Statistics")
                stats_display = gr.JSON(
                    label="Dataset Summary",
                    value={"status": "No datasets scanned yet"}
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Train/Test/Validation Split")

                gr.Markdown("**Industry Standard Ratios:** Train: 70%, Val: 15%, Test: 15%")

                train_ratio = gr.Slider(
                    minimum=0.5, maximum=0.9, value=0.7, step=0.05,
                    label="Train Ratio",
                    info="Training set ratio (70% recommended)"
                )
                val_ratio = gr.Slider(
                    minimum=0.05, maximum=0.3, value=0.15, step=0.05,
                    label="Validation Ratio",
                    info="Validation set ratio (15% recommended)"
                )
                test_ratio = gr.Slider(
                    minimum=0.05, maximum=0.3, value=0.15, step=0.05,
                    label="Test Ratio",
                    info="Test set ratio (15% recommended)"
                )

                split_button = gr.Button("✂️ Split Datasets", variant="primary")
                split_status = gr.Textbox(
                    label="Split Status",
                    value="Scan datasets first",
                    lines=2,
                    interactive=False
                )

            with gr.Column():
                gr.Markdown("### Health Report")
                health_report = gr.Textbox(
                    label="Dataset Health Analysis",
                    lines=15,
                    value="Run dataset scan to see health report...",
                    interactive=False
                )

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### .npy File Extraction")

        with gr.Row():
            with gr.Column():
                npy_folder = gr.Textbox(
                    label=".npy Files Directory",
                    placeholder="Path to .npy files (or leave empty to scan dataset_root/npy)",
                    lines=1
                )
                extract_button = gr.Button("📦 Extract .npy Files", variant="primary")

            with gr.Column():
                extraction_log = gr.Textbox(
                    label="Extraction Log",
                    lines=8,
                    value="Ready to extract .npy files...",
                    interactive=False
                )

        # Event handlers with full implementation
        def scan_datasets_handler(root_path: str, skip_val: bool, progress=gr.Progress()) -> Tuple[dict, str, str]:
            """Scan datasets and return statistics and health report"""
            global _current_scanner, _current_dataset_info

            try:
                if not root_path:
                    return (
                        {"error": "Please provide dataset root path"},
                        "❌ No path provided",
                        "Run dataset scan to see health report..."
                    )

                root_path = Path(root_path)
                if not root_path.exists():
                    return (
                        {"error": f"Path does not exist: {root_path}"},
                        f"❌ Directory not found: {root_path}",
                        "Run dataset scan to see health report..."
                    )

                logger.info(f"Scanning datasets in: {root_path} (skip_validation={skip_val})")

                # Progress callback for Gradio
                def update_progress(progress_value, message):
                    progress(progress_value, desc=f"Scanning: {message}")

                # Create scanner with caching and parallel processing
                use_cache = not skip_val  # Only use cache when validating
                scanner = DatasetScanner(root_path, use_cache=use_cache)

                # Scan datasets with progress
                progress(0, desc="Initializing scan...")
                dataset_info = scanner.scan_datasets(
                    progress_callback=update_progress,
                    skip_validation=skip_val
                )

                # Get statistics
                progress(0.95, desc="Generating statistics...")
                stats = scanner.get_statistics()

                # Save scanner for later use
                _current_scanner = scanner
                _current_dataset_info = dataset_info

                # Generate health report
                progress(0.97, desc="Generating health report...")
                health_checker = DatasetHealthChecker(stats)
                health_report_text = health_checker.generate_report()

                # Save manifest
                progress(0.99, desc="Saving manifest...")
                manifest_path = Path("data/splits/dataset_manifest.json")
                scanner.save_manifest(manifest_path)

                logger.info("Dataset scan complete")

                # Add cache info to status message
                cache_msg = ""
                if dataset_info.get('cached_files', 0) > 0:
                    cache_msg = f" ({dataset_info['cached_files']} from cache)"

                mode_msg = " (fast scan)" if skip_val else ""

                progress(1.0, desc="Complete!")

                return (
                    stats,
                    f"✅ Scan complete! Found {stats['total_files']} audio files{cache_msg}{mode_msg}",
                    health_report_text
                )

            except Exception as e:
                error_msg = f"Error scanning datasets: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    {"error": error_msg},
                    f"❌ Error: {str(e)}",
                    f"❌ Error during scan:\n{str(e)}"
                )

        def split_datasets_handler(
            root_path: str,
            train: float,
            val: float,
            test: float,
            progress=gr.Progress()
        ) -> Tuple[str, str]:
            """Split datasets into train/val/test"""
            global _current_dataset_info

            try:
                # Validate ratios
                total = train + val + test
                if abs(total - 1.0) > 0.01:
                    return (
                        f"❌ Ratios must sum to 1.0 (current: {total:.2f})",
                        "No split performed - fix ratios"
                    )

                if _current_dataset_info is None:
                    return (
                        "❌ Please scan datasets first",
                        "No split performed - scan required"
                    )

                logger.info(f"Splitting datasets: {train}/{val}/{test}")

                # Create splitter and split
                progress(0.1, desc="Initializing splitter...")
                splitter = DatasetSplitter(_current_dataset_info)

                progress(0.2, desc="Splitting datasets...")
                splits = splitter.split_datasets(
                    train_ratio=train,
                    val_ratio=val,
                    test_ratio=test,
                    random_seed=42,
                    stratify=True
                )

                # Save splits
                progress(0.7, desc="Saving splits...")
                output_dir = Path("data/splits")
                splitter.save_splits(output_dir)

                # Get split statistics
                progress(0.9, desc="Generating statistics...")
                split_stats = splitter.get_split_statistics()

                # Generate report
                report = ["=" * 60]
                report.append("DATASET SPLIT SUMMARY")
                report.append("=" * 60)
                report.append("")

                for split_name, stats in split_stats.items():
                    report.append(f"{split_name.upper()}:")
                    report.append(f"  Total Files: {stats['total_files']}")
                    report.append(f"  Percentage: {stats['percentage']:.1f}%")
                    report.append(f"  Categories: {stats['categories']}")
                    report.append("")

                report.append(f"✅ Splits saved to: {output_dir}")
                report.append("=" * 60)

                report_text = "\n".join(report)

                logger.info("Dataset split complete")

                progress(1.0, desc="Complete!")

                return (
                    f"✅ Split complete! Saved to {output_dir}",
                    report_text
                )

            except Exception as e:
                error_msg = f"Error splitting datasets: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    f"❌ Error: {str(e)}",
                    f"❌ Error during split:\n{str(e)}"
                )

        def extract_npy_handler(npy_path: str, root_path: str, progress=gr.Progress()) -> str:
            """Extract .npy files"""
            try:
                # Determine npy folder path
                if not npy_path:
                    if not root_path:
                        return "❌ Please provide either .npy folder path or scan datasets first"
                    npy_path = Path(root_path) / "npy"
                else:
                    npy_path = Path(npy_path)

                if not npy_path.exists():
                    return f"❌ Directory not found: {npy_path}"

                logger.info(f"Extracting .npy files from: {npy_path}")

                # Create extractor with parallel processing
                progress(0.05, desc="Initializing extractor...")
                extractor = NpyExtractor()

                # Scan for .npy files
                progress(0.1, desc="Scanning for .npy files...")
                npy_files = extractor.scan_npy_files(npy_path, recursive=True)

                if not npy_files:
                    return f"ℹ️  No .npy files found in: {npy_path}"

                # Progress callback for extraction
                def update_progress(current, total, message):
                    progress_value = 0.1 + (current / total) * 0.8
                    progress(progress_value, desc=f"Extracting: {message}")

                # Extract and analyze
                progress(0.1, desc=f"Processing {len(npy_files)} .npy files...")
                results = extractor.extract_and_convert(npy_files, progress_callback=update_progress)

                # Generate report
                progress(0.95, desc="Generating report...")
                report = extractor.generate_report()

                logger.info("NPY extraction complete")

                progress(1.0, desc="Complete!")

                return report

            except Exception as e:
                error_msg = f"Error extracting .npy files: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ Error: {str(e)}"

        # Connect event handlers
        scan_button.click(
            fn=scan_datasets_handler,
            inputs=[dataset_root, skip_validation],
            outputs=[stats_display, scan_status, health_report]
        )

        split_button.click(
            fn=split_datasets_handler,
            inputs=[dataset_root, train_ratio, val_ratio, test_ratio],
            outputs=[split_status, health_report]
        )

        extract_button.click(
            fn=extract_npy_handler,
            inputs=[npy_folder, dataset_root],
            outputs=[extraction_log]
        )

    return panel


if __name__ == "__main__":
    # Test the panel
    demo = create_dataset_panel()
    demo.launch()