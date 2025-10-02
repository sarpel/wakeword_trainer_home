"""
Panel 3: Model Training
- Start/pause/stop training with async execution
- Live metrics display
- Real-time plotting
- GPU monitoring
- Training state management
"""
import gradio as gr
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import time
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import logging

from src.config.defaults import WakewordConfig
from src.config.cuda_utils import get_cuda_validator
from src.data.dataset import load_dataset_splits
from src.models.architectures import create_model
from src.training.trainer import Trainer
from src.training.metrics import MetricResults

logger = logging.getLogger(__name__)


class TrainingState:
    """Global training state manager"""
    def __init__(self):
        self.is_training = False
        self.should_stop = False
        self.should_pause = False
        self.is_paused = False

        self.trainer: Optional[Trainer] = None
        self.config: Optional[WakewordConfig] = None
        self.model = None
        self.train_loader = None
        self.val_loader = None

        # Metrics history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_fpr': [],
            'val_fnr': [],
            'epochs': []
        }

        # Current metrics
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.current_train_loss = 0.0
        self.current_train_acc = 0.0
        self.current_val_loss = 0.0
        self.current_val_acc = 0.0
        self.current_fpr = 0.0
        self.current_fnr = 0.0
        self.current_speed = 0.0
        self.eta_seconds = 0

        # Best metrics
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_path = "No model saved yet"

        # Training thread
        self.training_thread = None
        self.log_queue = queue.Queue()

    def reset(self):
        """Reset state for new training"""
        self.is_training = False
        self.should_stop = False
        self.should_pause = False
        self.is_paused = False
        self.current_epoch = 0
        self.current_batch = 0
        self.eta_seconds = 0

    def add_log(self, message: str):
        """Add message to log queue"""
        self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] {message}\n")


# Global training state
training_state = TrainingState()


def create_loss_plot() -> plt.Figure:
    """Create loss curve plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(training_state.history['epochs']) > 0:
        epochs = training_state.history['epochs']

        ax.plot(epochs, training_state.history['train_loss'],
               label='Train Loss', marker='o', linewidth=2)
        ax.plot(epochs, training_state.history['val_loss'],
               label='Val Loss', marker='s', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_accuracy_plot() -> plt.Figure:
    """Create accuracy curve plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(training_state.history['epochs']) > 0:
        epochs = training_state.history['epochs']

        ax.plot(epochs, [a*100 for a in training_state.history['train_acc']],
               label='Train Acc', marker='o', linewidth=2)
        ax.plot(epochs, [a*100 for a in training_state.history['val_acc']],
               label='Val Acc', marker='s', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    else:
        ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_metrics_plot() -> plt.Figure:
    """Create FPR/FNR plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(training_state.history['epochs']) > 0:
        epochs = training_state.history['epochs']

        ax.plot(epochs, [f*100 for f in training_state.history['val_fpr']],
               label='FPR', marker='o', linewidth=2, color='red')
        ax.plot(epochs, [f*100 for f in training_state.history['val_fnr']],
               label='FNR', marker='s', linewidth=2, color='orange')
        ax.plot(epochs, [f*100 for f in training_state.history['val_f1']],
               label='F1 Score', marker='^', linewidth=2, color='green')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Rate (%)', fontsize=12)
        ax.set_title('Validation Metrics (FPR, FNR, F1)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    if seconds <= 0 or seconds == float('inf'):
        return "--:--:--"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def training_worker():
    """Background thread for training"""
    try:
        training_state.add_log("Starting training...")
        training_state.add_log(f"Configuration: {training_state.config.config_name}")
        training_state.add_log(f"Model: {training_state.config.model.architecture}")
        training_state.add_log(f"Epochs: {training_state.config.training.epochs}")
        training_state.add_log(f"Batch size: {training_state.config.training.batch_size}")
        training_state.add_log("-" * 60)

        # Create custom callback for live updates
        class LiveUpdateCallback:
            def on_epoch_end(self, epoch, train_loss, val_loss, val_metrics: MetricResults):
                # Update state
                training_state.current_epoch = epoch + 1
                training_state.current_train_loss = train_loss
                training_state.current_val_loss = val_loss
                training_state.current_val_acc = val_metrics.accuracy
                training_state.current_fpr = val_metrics.fpr
                training_state.current_fnr = val_metrics.fnr

                # Update history
                training_state.history['epochs'].append(epoch + 1)
                training_state.history['train_loss'].append(train_loss)
                training_state.history['train_acc'].append(training_state.current_train_acc)
                training_state.history['val_loss'].append(val_loss)
                training_state.history['val_acc'].append(val_metrics.accuracy)
                training_state.history['val_f1'].append(val_metrics.f1_score)
                training_state.history['val_fpr'].append(val_metrics.fpr)
                training_state.history['val_fnr'].append(val_metrics.fnr)

                # Update best metrics
                if val_loss < training_state.best_val_loss:
                    training_state.best_val_loss = val_loss
                    training_state.best_epoch = epoch + 1
                    training_state.best_model_path = str(
                        training_state.trainer.checkpoint_dir / "best_model.pt"
                    )

                if val_metrics.accuracy > training_state.best_val_acc:
                    training_state.best_val_acc = val_metrics.accuracy

                # Log
                training_state.add_log(
                    f"Epoch {epoch+1}/{training_state.total_epochs} - "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                    f"Acc: {training_state.current_train_acc:.2%}/{val_metrics.accuracy:.2%} - "
                    f"FPR: {val_metrics.fpr:.2%} - FNR: {val_metrics.fnr:.2%}"
                )

            def on_batch_end(self, batch_idx, loss, acc):
                training_state.current_batch = batch_idx + 1
                training_state.current_train_loss = loss
                training_state.current_train_acc = acc

        # Add callback
        callback = LiveUpdateCallback()
        training_state.trainer.add_callback(callback)

        # Train
        start_time = time.time()
        results = training_state.trainer.train()
        elapsed = time.time() - start_time

        # Training complete
        training_state.add_log("-" * 60)
        training_state.add_log(f"Training complete!")
        training_state.add_log(f"Total time: {elapsed/3600:.2f} hours")
        training_state.add_log(f"Best epoch: {training_state.best_epoch}")
        training_state.add_log(f"Best val loss: {training_state.best_val_loss:.4f}")
        training_state.add_log(f"Best val acc: {training_state.best_val_acc:.2%}")
        training_state.add_log(f"Model saved to: {training_state.best_model_path}")

    except Exception as e:
        training_state.add_log(f"ERROR: {str(e)}")
        logger.exception("Training failed")
    finally:
        training_state.is_training = False


def start_training(config_state: Dict) -> Tuple:
    """Start training with current configuration"""
    if training_state.is_training:
        return (
            "⚠️ Training already in progress",
            f"{training_state.current_epoch}/{training_state.total_epochs}",
            f"{training_state.current_batch}/{training_state.total_batches}",
            None, None, None
        )

    try:
        # Get config from global state
        if 'config' not in config_state or config_state['config'] is None:
            return (
                "❌ No configuration loaded. Please configure in Panel 2 first.",
                "0/0", "0/0", None, None, None
            )

        config = config_state['config']
        training_state.config = config
        training_state.total_epochs = config.training.epochs

        training_state.add_log("Initializing training...")

        # Check if dataset splits exist
        splits_dir = Path("data/splits")
        if not splits_dir.exists() or not (splits_dir / "train.json").exists():
            return (
                "❌ Dataset splits not found. Please run Panel 1 to scan and split datasets first.",
                "0/0", "0/0", None, None, None
            )

        training_state.add_log("Loading datasets...")

        # Load datasets
        aug_config = {
            'time_stretch_range': (config.augmentation.time_stretch_min,
                                  config.augmentation.time_stretch_max),
            'pitch_shift_range': (config.augmentation.pitch_shift_min,
                                 config.augmentation.pitch_shift_max),
            'background_noise_prob': config.augmentation.background_noise_prob,
            'noise_snr_range': (config.augmentation.noise_snr_min,
                               config.augmentation.noise_snr_max),
            'rir_prob': config.augmentation.rir_prob
        }

        # Normalize feature type name
        feature_type = 'mel' if config.data.feature_type == 'mel_spectrogram' else config.data.feature_type

        train_ds, val_ds, test_ds = load_dataset_splits(
            splits_dir=splits_dir,
            sample_rate=config.data.sample_rate,
            audio_duration=config.data.audio_duration,
            augment_train=True,
            augmentation_config=aug_config,
            data_root=Path("data"),
            device='cuda',
            feature_type=feature_type,
            n_mels=config.data.n_mels,
            n_mfcc=config.data.n_mfcc,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length
        )

        training_state.add_log(f"Loaded {len(train_ds)} training samples")
        training_state.add_log(f"Loaded {len(val_ds)} validation samples")

        # Create data loaders
        training_state.train_loader = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=True,
            persistent_workers=True if config.training.num_workers > 0 else False
        )

        training_state.val_loader = DataLoader(
            val_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
            persistent_workers=True if config.training.num_workers > 0 else False
        )

        training_state.total_batches = len(training_state.train_loader)

        training_state.add_log("Creating model...")

        # Create model
        training_state.model = create_model(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout
        )

        training_state.add_log(f"Model created: {config.model.architecture}")

        # Create checkpoint directory
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        training_state.add_log("Initializing trainer...")

        # Create trainer
        training_state.trainer = Trainer(
            model=training_state.model,
            train_loader=training_state.train_loader,
            val_loader=training_state.val_loader,
            config=config,
            checkpoint_dir=checkpoint_dir,
            device='cuda'
        )

        # Reset history
        training_state.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_fpr': [],
            'val_fnr': [],
            'epochs': []
        }

        # Start training in background thread
        training_state.is_training = True
        training_state.should_stop = False
        training_state.training_thread = threading.Thread(target=training_worker, daemon=True)
        training_state.training_thread.start()

        return (
            "✅ Training started!",
            f"0/{training_state.total_epochs}",
            f"0/{training_state.total_batches}",
            create_loss_plot(),
            create_accuracy_plot(),
            create_metrics_plot()
        )

    except Exception as e:
        error_msg = f"❌ Failed to start training: {str(e)}"
        training_state.add_log(error_msg)
        logger.exception("Failed to start training")
        return (
            error_msg,
            "0/0", "0/0", None, None, None
        )


def stop_training() -> str:
    """Stop training"""
    if not training_state.is_training:
        return "⚠️ No training in progress"

    training_state.should_stop = True
    training_state.add_log("Stop requested. Training will stop after current epoch...")

    return "⏹️ Stopping training..."


def get_training_status() -> Tuple:
    """Get current training status for live updates"""
    # Close old matplotlib figures to prevent memory leak
    plt.close('all')

    # Collect logs
    logs = ""
    while not training_state.log_queue.empty():
        try:
            logs += training_state.log_queue.get_nowait()
        except queue.Empty:
            break

    # Status message
    if training_state.is_training:
        status = f"🔄 Training in progress (Epoch {training_state.current_epoch}/{training_state.total_epochs})"
    else:
        status = "✅ Ready to train"

    # Calculate ETA (simple estimation)
    if training_state.is_training and training_state.current_epoch > 0:
        # Rough estimate based on current progress
        epochs_remaining = training_state.total_epochs - training_state.current_epoch
        # Assume similar time per epoch
        training_state.eta_seconds = epochs_remaining * 60  # Placeholder
    else:
        training_state.eta_seconds = 0

    # GPU utilization
    try:
        validator = get_cuda_validator()
        gpu_util = validator.get_memory_info()
        gpu_percent = gpu_util['allocated_gb'] / gpu_util['total_gb'] * 100
    except:
        gpu_percent = 0.0

    return (
        status,
        f"{training_state.current_epoch}/{training_state.total_epochs}",
        f"{training_state.current_batch}/{training_state.total_batches}",
        round(training_state.current_train_loss, 4),
        round(training_state.current_val_loss, 4),
        round(training_state.current_train_acc * 100, 2),
        round(training_state.current_val_acc * 100, 2),
        round(training_state.current_fpr * 100, 2),
        round(training_state.current_fnr * 100, 2),
        round(training_state.current_speed, 1),
        round(gpu_percent, 1),
        format_time(training_state.eta_seconds),
        logs,
        create_loss_plot(),
        create_accuracy_plot(),
        create_metrics_plot(),
        str(training_state.best_epoch),
        round(training_state.best_val_loss, 4),
        round(training_state.best_val_acc * 100, 2),
        training_state.best_model_path
    )


def create_training_panel(state: gr.State) -> gr.Blocks:
    """
    Create Panel 3: Model Training

    Args:
        state: Global state dictionary

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 🚀 Model Training")
        gr.Markdown("Train your wakeword model with real-time monitoring and GPU acceleration.")

        with gr.Row():
            start_training_btn = gr.Button("▶️ Start Training", variant="primary", scale=2)
            stop_training_btn = gr.Button("⏹️ Stop Training", variant="stop", scale=1)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Training Status")
                training_status = gr.Textbox(
                    label="Status",
                    value="Ready to train",
                    lines=2,
                    interactive=False
                )

                gr.Markdown("### Current Progress")
                current_epoch = gr.Textbox(label="Epoch", value="0/0", interactive=False)
                current_batch = gr.Textbox(label="Batch", value="0/0", interactive=False)

                gr.Markdown("### Current Metrics")
                with gr.Row():
                    train_loss = gr.Number(label="Train Loss", value=0.0, interactive=False)
                    val_loss = gr.Number(label="Val Loss", value=0.0, interactive=False)

                with gr.Row():
                    train_acc = gr.Number(label="Train Acc (%)", value=0.0, interactive=False)
                    val_acc = gr.Number(label="Val Acc (%)", value=0.0, interactive=False)

                with gr.Row():
                    fpr = gr.Number(label="FPR (%)", value=0.0, interactive=False)
                    fnr = gr.Number(label="FNR (%)", value=0.0, interactive=False)

                with gr.Row():
                    speed = gr.Number(label="Speed (samples/sec)", value=0.0, interactive=False)
                    gpu_util = gr.Number(label="GPU Util (%)", value=0.0, interactive=False)

                eta = gr.Textbox(label="ETA", value="--:--:--", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Training Curves")

                # Loss plot
                loss_plot = gr.Plot(
                    label="Loss Curves",
                    value=create_loss_plot()
                )

                # Accuracy plot
                accuracy_plot = gr.Plot(
                    label="Accuracy Curves",
                    value=create_accuracy_plot()
                )

                # Metrics plot
                metrics_plot = gr.Plot(
                    label="Validation Metrics (FPR, FNR, F1)",
                    value=create_metrics_plot()
                )

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### Training Log")

        with gr.Row():
            training_log = gr.Textbox(
                label="Console Output",
                lines=8,
                value="Waiting to start training...\n",
                interactive=False,
                max_lines=100,
                autoscroll=True
            )

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### Best Model Info")

        with gr.Row():
            with gr.Column():
                best_epoch = gr.Textbox(label="Best Epoch", value="--", interactive=False)
                best_val_loss = gr.Number(label="Best Val Loss", value=0.0, interactive=False)
            with gr.Column():
                best_val_acc = gr.Number(label="Best Val Acc (%)", value=0.0, interactive=False)
                model_path = gr.Textbox(
                    label="Checkpoint Path",
                    value="No model saved yet",
                    interactive=False
                )

        # Event handlers
        start_training_btn.click(
            fn=start_training,
            inputs=[state],
            outputs=[
                training_status,
                current_epoch,
                current_batch,
                loss_plot,
                accuracy_plot,
                metrics_plot
            ]
        )

        stop_training_btn.click(
            fn=stop_training,
            outputs=[training_status]
        )

        # Auto-refresh for live updates
        status_refresh = gr.Timer(value=2.0, active=True)  # Update every 2 seconds

        status_refresh.tick(
            fn=get_training_status,
            outputs=[
                training_status,
                current_epoch,
                current_batch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                fpr,
                fnr,
                speed,
                gpu_util,
                eta,
                training_log,
                loss_plot,
                accuracy_plot,
                metrics_plot,
                best_epoch,
                best_val_loss,
                best_val_acc,
                model_path
            ]
        )

    return panel


if __name__ == "__main__":
    # Test the panel
    state = gr.State(value={})
    demo = create_training_panel(state)
    demo.launch()
