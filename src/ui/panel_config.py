"""
Panel 2: Configuration Management
- Basic and advanced parameter editing
- Configuration presets
- Save/load configuration
"""
import gradio as gr
from pathlib import Path
from typing import Tuple, List
import sys
import traceback
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.defaults import WakewordConfig
from src.config.presets import get_preset, list_presets
from src.config.validator import ConfigValidator
from src.config.logger import get_logger

logger = get_logger("config")

# Global state for current configuration
_current_config = None


def create_config_panel(state: gr.State = None) -> gr.Blocks:
    """
    Create Panel 2: Configuration Management

    Args:
        state: Global state dictionary for sharing config between panels

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# ⚙️ Training Configuration")
        gr.Markdown("Configure all training parameters - basic and advanced.")

        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list_presets(),
                value="Default",
                label="Configuration Preset",
                info="Select a preset optimized for your use case"
            )
            load_preset_btn = gr.Button("📥 Load Preset", variant="secondary")

        with gr.Row():
            save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
            load_config_btn = gr.Button("📂 Load Configuration", variant="secondary")
            reset_config_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
            validate_btn = gr.Button("✅ Validate Configuration", variant="secondary")

        with gr.Tabs():
            # Basic Configuration Tab
            with gr.TabItem("Basic Parameters"):
                gr.Markdown("### Data Parameters")
                with gr.Row():
                    sample_rate = gr.Number(
                        label="Sample Rate (Hz)",
                        value=16000,
                        info="Audio sample rate (16kHz recommended)"
                    )
                    audio_duration = gr.Number(
                        label="Audio Duration (seconds)",
                        value=1.5,
                        info="Length of audio clips (1.5-2s typical)"
                    )

                with gr.Row():
                    n_mfcc = gr.Slider(
                        minimum=13, maximum=80, value=40, step=1,
                        label="MFCC Coefficients",
                        info="Number of MFCC features"
                    )
                    n_fft = gr.Dropdown(
                        choices=[256, 512, 1024, 2048],
                        value=512,
                        label="FFT Size",
                        info="FFT window size"
                    )

                gr.Markdown("### Training Parameters")
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=8, maximum=256, value=32, step=8,
                        label="Batch Size",
                        info="Training batch size (GPU memory dependent)"
                    )
                    epochs = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10,
                        label="Epochs",
                        info="Number of training epochs"
                    )

                with gr.Row():
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=0.001,
                        info="Initial learning rate (0.001 recommended)"
                    )
                    early_stopping = gr.Slider(
                        minimum=5, maximum=30, value=10, step=5,
                        label="Early Stopping Patience",
                        info="Epochs to wait before stopping"
                    )

                gr.Markdown("### Model Parameters")
                with gr.Row():
                    architecture = gr.Dropdown(
                        choices=["resnet18", "mobilenetv3", "lstm", "gru", "tcn"],
                        value="resnet18",
                        label="Model Architecture",
                        info="ResNet18 recommended for accuracy"
                    )
                    num_classes = gr.Number(
                        label="Number of Classes",
                        value=2,
                        info="Binary classification (2)"
                    )

            # Advanced Configuration Tab
            with gr.TabItem("Advanced Parameters"):
                gr.Markdown("### Augmentation")
                with gr.Row():
                    time_stretch_min = gr.Number(
                        label="Time Stretch Min",
                        value=0.8,
                        info="Minimum time stretch rate"
                    )
                    time_stretch_max = gr.Number(
                        label="Time Stretch Max",
                        value=1.2,
                        info="Maximum time stretch rate"
                    )

                with gr.Row():
                    pitch_shift_min = gr.Number(
                        label="Pitch Shift Min (semitones)",
                        value=-2
                    )
                    pitch_shift_max = gr.Number(
                        label="Pitch Shift Max (semitones)",
                        value=2
                    )

                with gr.Row():
                    background_noise_prob = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.1,
                        label="Background Noise Probability"
                    )
                    rir_prob = gr.Slider(
                        minimum=0, maximum=1, value=0.3, step=0.1,
                        label="RIR Probability"
                    )

                with gr.Row():
                    noise_snr_min = gr.Number(
                        label="Noise SNR Min (dB)",
                        value=5
                    )
                    noise_snr_max = gr.Number(
                        label="Noise SNR Max (dB)",
                        value=20
                    )

                gr.Markdown("### Optimizer & Scheduler")
                with gr.Row():
                    optimizer = gr.Dropdown(
                        choices=["adam", "sgd", "adamw"],
                        value="adam",
                        label="Optimizer"
                    )
                    scheduler = gr.Dropdown(
                        choices=["cosine", "step", "plateau", "none"],
                        value="cosine",
                        label="Learning Rate Scheduler"
                    )

                with gr.Row():
                    weight_decay = gr.Number(
                        label="Weight Decay",
                        value=1e-4
                    )
                    gradient_clip = gr.Number(
                        label="Gradient Clipping",
                        value=1.0
                    )

                with gr.Row():
                    mixed_precision = gr.Checkbox(
                        label="Mixed Precision Training (FP16)",
                        value=True,
                        info="Faster training, less memory"
                    )
                    num_workers = gr.Slider(
                        minimum=0, maximum=16, value=4, step=1,
                        label="Data Loader Workers"
                    )

                gr.Markdown("### Loss & Sampling")
                with gr.Row():
                    loss_function = gr.Dropdown(
                        choices=["cross_entropy", "focal_loss"],
                        value="cross_entropy",
                        label="Loss Function"
                    )
                    label_smoothing = gr.Slider(
                        minimum=0, maximum=0.3, value=0.1, step=0.05,
                        label="Label Smoothing"
                    )

                with gr.Row():
                    class_weights = gr.Dropdown(
                        choices=["balanced", "none", "custom"],
                        value="balanced",
                        label="Class Weights"
                    )
                    hard_negative_weight = gr.Number(
                        label="Hard Negative Weight",
                        value=2.0
                    )

                gr.Markdown("### Checkpointing")
                with gr.Row():
                    checkpoint_frequency = gr.Dropdown(
                        choices=["every_epoch", "every_5_epochs", "every_10_epochs", "best_only"],
                        value="best_only",
                        label="Checkpoint Frequency"
                    )

        gr.Markdown("---")

        with gr.Row():
            config_status = gr.Textbox(
                label="Configuration Status",
                value="Configuration ready. Modify parameters above or load a preset.",
                lines=3,
                interactive=False
            )

        with gr.Row():
            validation_report = gr.Textbox(
                label="Validation Report",
                value="Click 'Validate Configuration' to check parameters.",
                lines=10,
                interactive=False,
                visible=False
            )

        # Collect all inputs for easier handling
        all_inputs = [
            # Data
            sample_rate, audio_duration, n_mfcc, n_fft,
            # Training
            batch_size, epochs, learning_rate, early_stopping,
            # Model
            architecture, num_classes,
            # Augmentation
            time_stretch_min, time_stretch_max,
            pitch_shift_min, pitch_shift_max,
            background_noise_prob, rir_prob,
            noise_snr_min, noise_snr_max,
            # Optimizer
            optimizer, scheduler, weight_decay, gradient_clip,
            mixed_precision, num_workers,
            # Loss
            loss_function, label_smoothing,
            class_weights, hard_negative_weight,
            # Checkpointing
            checkpoint_frequency
        ]

        # Event handlers with full implementation
        def _params_to_config(params):
            """Convert UI parameters to WakewordConfig"""
            from src.config.defaults import (
                DataConfig, TrainingConfig, ModelConfig,
                AugmentationConfig, OptimizerConfig, LossConfig
            )

            return WakewordConfig(
                config_name="custom",
                description="Custom configuration from UI",
                data=DataConfig(
                    sample_rate=int(params[0]),
                    audio_duration=float(params[1]),
                    n_mfcc=int(params[2]),
                    n_fft=int(params[3])
                ),
                training=TrainingConfig(
                    batch_size=int(params[4]),
                    epochs=int(params[5]),
                    learning_rate=float(params[6]),
                    early_stopping_patience=int(params[7]),
                    num_workers=int(params[21]),
                    checkpoint_frequency=params[28]
                ),
                model=ModelConfig(
                    architecture=params[8],
                    num_classes=int(params[9])
                ),
                augmentation=AugmentationConfig(
                    time_stretch_min=float(params[10]),
                    time_stretch_max=float(params[11]),
                    pitch_shift_min=int(params[12]),
                    pitch_shift_max=int(params[13]),
                    background_noise_prob=float(params[14]),
                    rir_prob=float(params[15]),
                    noise_snr_min=float(params[16]),
                    noise_snr_max=float(params[17])
                ),
                optimizer=OptimizerConfig(
                    optimizer=params[18],
                    scheduler=params[19],
                    weight_decay=float(params[20]),
                    gradient_clip=float(params[21]),
                    mixed_precision=bool(params[22])
                ),
                loss=LossConfig(
                    loss_function=params[24],
                    label_smoothing=float(params[25]),
                    class_weights=params[26],
                    hard_negative_weight=float(params[27])
                )
            )

        def _config_to_params(config: WakewordConfig) -> List:
            """Convert WakewordConfig to UI parameters"""
            return [
                # Data
                config.data.sample_rate,
                config.data.audio_duration,
                config.data.n_mfcc,
                config.data.n_fft,
                # Training
                config.training.batch_size,
                config.training.epochs,
                config.training.learning_rate,
                config.training.early_stopping_patience,
                # Model
                config.model.architecture,
                config.model.num_classes,
                # Augmentation
                config.augmentation.time_stretch_min,
                config.augmentation.time_stretch_max,
                config.augmentation.pitch_shift_min,
                config.augmentation.pitch_shift_max,
                config.augmentation.background_noise_prob,
                config.augmentation.rir_prob,
                config.augmentation.noise_snr_min,
                config.augmentation.noise_snr_max,
                # Optimizer
                config.optimizer.optimizer,
                config.optimizer.scheduler,
                config.optimizer.weight_decay,
                config.optimizer.gradient_clip,
                config.optimizer.mixed_precision,
                config.training.num_workers,
                # Loss
                config.loss.loss_function,
                config.loss.label_smoothing,
                config.loss.class_weights,
                config.loss.hard_negative_weight,
                # Checkpointing
                config.training.checkpoint_frequency
            ]

        def load_preset_handler(preset_name: str) -> Tuple:
            """Load configuration preset"""
            global _current_config

            try:
                logger.info(f"Loading preset: {preset_name}")

                # Get preset configuration
                config = get_preset(preset_name)
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value['config'] = config

                # Convert to UI parameters
                params = _config_to_params(config)

                # Prepare status message
                status = f"✅ Loaded preset: {preset_name}\n{config.description}"

                logger.info(f"Preset loaded successfully: {preset_name}")

                return tuple(params + [status, gr.update(visible=False)])

            except Exception as e:
                error_msg = f"Error loading preset: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return tuple([None] * len(all_inputs) + [f"❌ {error_msg}", gr.update(visible=False)])

        def save_config_handler(*params) -> str:
            """Save configuration to YAML file"""
            global _current_config

            try:
                # Create config from current parameters
                config = _params_to_config(params)
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value['config'] = config

                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(f"configs/config_{timestamp}.yaml")
                config.save(save_path)

                logger.info(f"Configuration saved to: {save_path}")

                return f"✅ Configuration saved to: {save_path}"

            except Exception as e:
                error_msg = f"Error saving configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ {error_msg}"

        def load_config_handler() -> Tuple:
            """Load configuration from YAML file"""
            global _current_config

            try:
                # Find most recent config
                config_dir = Path("configs")
                if not config_dir.exists():
                    return tuple([None] * len(all_inputs) + ["❌ No saved configurations found", gr.update(visible=False)])

                config_files = sorted(config_dir.glob("config_*.yaml"), reverse=True)
                if not config_files:
                    return tuple([None] * len(all_inputs) + ["❌ No saved configurations found", gr.update(visible=False)])

                # Load most recent
                latest_config = config_files[0]
                logger.info(f"Loading configuration from: {latest_config}")

                config = WakewordConfig.load(latest_config)
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value['config'] = config

                # Convert to UI parameters
                params = _config_to_params(config)

                status = f"✅ Loaded configuration from: {latest_config.name}"
                logger.info("Configuration loaded successfully")

                return tuple(params + [status, gr.update(visible=False)])

            except Exception as e:
                error_msg = f"Error loading configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return tuple([None] * len(all_inputs) + [f"❌ {error_msg}", gr.update(visible=False)])

        def reset_config_handler() -> Tuple:
            """Reset to default configuration"""
            global _current_config

            try:
                logger.info("Resetting to default configuration")

                # Get default config
                config = get_preset("Default")
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value['config'] = config

                # Convert to UI parameters
                params = _config_to_params(config)

                status = "✅ Reset to default configuration"
                logger.info("Configuration reset successfully")

                return tuple(params + [status, gr.update(visible=False)])

            except Exception as e:
                error_msg = f"Error resetting configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return tuple([None] * len(all_inputs) + [f"❌ {error_msg}", gr.update(visible=False)])

        def validate_config_handler(*params) -> Tuple[str, str]:
            """Validate current configuration"""
            try:
                logger.info("Validating configuration")

                # Create config from current parameters
                config = _params_to_config(params)

                # Validate
                validator = ConfigValidator()
                is_valid, issues = validator.validate(config)

                # Generate report
                report = validator.generate_report()

                if is_valid:
                    status = "✅ Configuration is valid and ready for training"
                else:
                    status = f"❌ Configuration has {len([i for i in issues if i.severity == 'error'])} errors"

                logger.info(f"Validation complete: {'valid' if is_valid else 'invalid'}")

                return status, gr.update(value=report, visible=True)

            except Exception as e:
                error_msg = f"Error validating configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ {error_msg}", gr.update(visible=False)

        # Connect event handlers
        load_preset_btn.click(
            fn=load_preset_handler,
            inputs=[preset_dropdown],
            outputs=all_inputs + [config_status, validation_report]
        )

        save_config_btn.click(
            fn=save_config_handler,
            inputs=all_inputs,
            outputs=[config_status]
        )

        load_config_btn.click(
            fn=load_config_handler,
            outputs=all_inputs + [config_status, validation_report]
        )

        reset_config_btn.click(
            fn=reset_config_handler,
            outputs=all_inputs + [config_status, validation_report]
        )

        validate_btn.click(
            fn=validate_config_handler,
            inputs=all_inputs,
            outputs=[config_status, validation_report]
        )

    return panel


if __name__ == "__main__":
    # Test the panel
    demo = create_config_panel()
    demo.launch()