"""
Wakeword Training Loop
GPU-accelerated training with checkpointing, early stopping, and metrics tracking
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

from src.training.metrics import MetricsTracker, MetricMonitor, MetricResults
from src.training.optimizer_factory import (
    create_optimizer_and_scheduler,
    create_grad_scaler,
    clip_gradients,
    get_learning_rate
)
from src.models.losses import create_loss_function
from src.config.cuda_utils import enforce_cuda

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Container for training state"""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_val_f1: float = 0.0
    best_val_fpr: float = 1.0
    epochs_without_improvement: int = 0
    training_time: float = 0.0


class Trainer:
    """
    Main training loop for wakeword detection
    GPU-accelerated with comprehensive metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        checkpoint_dir: Optional[Path] = None,
        device: str = 'cuda'
    ):
        """
        Initialize trainer
        """
        # Enforce GPU requirement
        enforce_cuda()

        self.device = device
        self.config = config

        # Move model to GPU
        self.model = model.to(device)
        # channels_last bellek düzeni (Ampere+ için throughput ↑)
        self.model = self.model.to(memory_format=torch.channels_last)  # CHANGE

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Create loss function
        self.criterion = create_loss_function(
            loss_name=config.loss.loss_function,
            num_classes=config.model.num_classes,
            label_smoothing=config.loss.label_smoothing,
            focal_alpha=config.loss.focal_alpha,
            focal_gamma=config.loss.focal_gamma,
            class_weights=None,
            device=device
        ).to(device)

        # Create optimizer and scheduler (self.model ile kur)
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(self.model, config)  # CHANGE

        # Mixed precision training
        self.use_mixed_precision = config.optimizer.mixed_precision
        self.scaler = create_grad_scaler(enabled=self.use_mixed_precision)

        # Gradient clipping
        self.gradient_clip = config.optimizer.gradient_clip

        # Metrics tracking
        self.train_metrics_tracker = MetricsTracker(device=device)
        self.val_metrics_tracker = MetricsTracker(device=device)
        self.metric_monitor = MetricMonitor(window_size=100)

        # Training state
        self.state = TrainingState()

        # Early stopping
        self.early_stopping_patience = config.training.early_stopping_patience

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_frequency = config.training.checkpoint_frequency

        # Callbacks
        self.callbacks = []

        logger.info("Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model: {config.model.architecture}")
        logger.info(f"  Optimizer: {config.optimizer.optimizer}")
        logger.info(f"  Scheduler: {config.optimizer.scheduler}")
        logger.info(f"  Loss: {config.loss.loss_function}")
        logger.info(f"  Mixed precision: {self.use_mixed_precision}")
        logger.info(f"  Gradient clipping: {self.gradient_clip}")
        logger.info(f"  Early stopping patience: {self.early_stopping_patience}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics_tracker.reset()

        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # BUGFIX: Prevent division by zero if loader is empty
        if num_batches == 0:
            logger.warning("Training loader is empty, skipping epoch")
            return 0.0, 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.training.epochs} [Train]",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            try:
                # Unpack batch - BUGFIX: Validate batch structure
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    logger.error(f"Invalid batch structure at index {batch_idx}, skipping")
                    continue
                    
                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch

                # BUGFIX: Validate tensor shapes before processing
                if inputs.numel() == 0 or targets.numel() == 0:
                    logger.warning(f"Empty tensor in batch {batch_idx}, skipping")
                    continue

                # Move to device with channels_last
                inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # BUGFIX: Check for NaN/Inf loss before backprop
                if not torch.isfinite(loss):
                    logger.error(f"Non-finite loss detected at batch {batch_idx}: {loss.item()}")
                    continue

                # Backward with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = clip_gradients(self.model, self.gradient_clip)
                else:
                    grad_norm = 0.0

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Metrics
                with torch.no_grad():
                    pred_classes = torch.argmax(outputs, dim=1)
                    batch_acc = (pred_classes == targets).float().mean().item()

                self.train_metrics_tracker.update(outputs.detach(), targets.detach())
                self.metric_monitor.update_batch(loss.item(), batch_acc)

                epoch_loss += loss.item()
                running_avg = self.metric_monitor.get_running_averages()
                pbar.set_postfix({
                    'loss': f"{running_avg['loss']:.4f}",
                    'acc': f"{running_avg['accuracy']:.4f}",
                    'lr': f"{get_learning_rate(self.optimizer):.6f}"
                })

                self.state.global_step += 1
                self._call_callbacks('on_batch_end', batch_idx, loss.item(), batch_acc)
            
            except RuntimeError as e:
                # BUGFIX: Handle CUDA OOM and other runtime errors gracefully
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at batch {batch_idx}. Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    logger.exception(f"Runtime error at batch {batch_idx}: {e}")
                    raise  # Re-raise non-OOM errors
            except Exception as e:
                # BUGFIX: Catch all other exceptions to prevent training crash
                logger.exception(f"Unexpected error at batch {batch_idx}: {e}")
                continue

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        train_metrics = self.train_metrics_tracker.compute()

        logger.info(f"Epoch {epoch+1} [Train]: Loss={avg_loss:.4f}, {train_metrics}")

        return avg_loss, train_metrics.accuracy

    def validate_epoch(self, epoch: int) -> Tuple[float, MetricResults]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics_tracker.reset()

        epoch_loss = 0.0
        num_batches = len(self.val_loader)
        
        # BUGFIX: Handle empty validation loader
        if num_batches == 0:
            logger.warning("Validation loader is empty")
            return 0.0, MetricResults(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                fpr=0.0, fnr=0.0, true_positives=0, true_negatives=0,
                false_positives=0, false_negatives=0, total_samples=0,
                positive_samples=0, negative_samples=0
            )

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.config.training.epochs} [Val]",
            leave=False
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    # BUGFIX: Validate batch structure
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        logger.error(f"Invalid batch structure at validation batch {batch_idx}")
                        continue
                        
                    if len(batch) == 3:
                        inputs, targets, _ = batch
                    else:
                        inputs, targets = batch
                    
                    # BUGFIX: Validate tensors are not empty
                    if inputs.numel() == 0 or targets.numel() == 0:
                        logger.warning(f"Empty tensor in validation batch {batch_idx}")
                        continue

                    # channels_last tutarlılığı
                    inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)  # CHANGE
                    targets = targets.to(self.device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    
                    # BUGFIX: Check for NaN/Inf in validation loss
                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite validation loss at batch {batch_idx}")
                        continue

                    self.val_metrics_tracker.update(outputs.detach(), targets.detach())

                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                except RuntimeError as e:
                    # BUGFIX: Handle validation errors gracefully
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA OOM during validation at batch {batch_idx}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        logger.exception(f"Runtime error during validation at batch {batch_idx}: {e}")
                        raise
                except Exception as e:
                    logger.exception(f"Unexpected error during validation at batch {batch_idx}: {e}")
                    continue

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        val_metrics = self.val_metrics_tracker.compute()

        logger.info(f"Epoch {epoch+1} [Val]: Loss={avg_loss:.4f}, {val_metrics}")

        return avg_loss, val_metrics

    def train(
        self,
        start_epoch: int = 0,
        resume_from: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Full training loop"""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.state.epoch + 1
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_fpr': [],
            'val_fnr': [],
            'learning_rates': []
        }

        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info(f"  Epochs: {self.config.training.epochs}")
        logger.info(f"  Training batches: {len(self.train_loader)}")
        logger.info(f"  Validation batches: {len(self.val_loader)}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            for epoch in range(start_epoch, self.config.training.epochs):
                self.state.epoch = epoch
                self._call_callbacks('on_epoch_start', epoch)

                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_metrics = self.validate_epoch(epoch)

                self._update_scheduler(val_loss)
                current_lr = get_learning_rate(self.optimizer)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_metrics.accuracy)
                history['val_f1'].append(val_metrics.f1_score)
                history['val_fpr'].append(val_metrics.fpr)
                history['val_fnr'].append(val_metrics.fnr)
                history['learning_rates'].append(current_lr)

                self.val_metrics_tracker.save_epoch_metrics(val_metrics)

                improved = self._check_improvement(val_loss, val_metrics.f1_score, val_metrics.fpr)

                self._save_checkpoint(epoch, val_loss, val_metrics, improved)

                if self._should_stop_early():
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

                self._call_callbacks('on_epoch_end', epoch, train_loss, val_loss, val_metrics)

                print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
                print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_metrics.accuracy:.4f}, "
                      f"F1={val_metrics.f1_score:.4f}, FPR={val_metrics.fpr:.4f}, FNR={val_metrics.fnr:.4f}")
                print(f"  LR: {current_lr:.6f}")
                if improved:
                    print(f"  ✅ New best model (improvement detected)\n")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        end_time = time.time()
        self.state.training_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("Training complete")
        logger.info(f"  Total time: {self.state.training_time / 3600:.2f} hours")
        logger.info(f"  Best val loss: {self.state.best_val_loss:.4f}")
        logger.info(f"  Best val F1: {self.state.best_val_f1:.4f}")
        logger.info(f"  Best val FPR: {self.state.best_val_fpr:.4f}")
        logger.info("=" * 80)

        best_f1_epoch, best_f1_metrics = self.val_metrics_tracker.get_best_epoch('f1_score')
        best_fpr_epoch, best_fpr_metrics = self.val_metrics_tracker.get_best_epoch('fpr')

        logger.info(f"\nBest F1 Score: {best_f1_metrics.f1_score:.4f} (Epoch {best_f1_epoch+1})")
        logger.info(f"Best FPR: {best_fpr_metrics.fpr:.4f} (Epoch {best_fpr_epoch+1})")

        results = {
            'history': history,
            'final_epoch': self.state.epoch,
            'best_val_loss': self.state.best_val_loss,
            'best_val_f1': self.state.best_val_f1,
            'best_val_fpr': self.state.best_val_fpr,
            'training_time': self.state.training_time,
            'best_f1_epoch': best_f1_epoch,
            'best_fpr_epoch': best_fpr_epoch
        }

        return results

    def _update_scheduler(self, val_loss: float):
        """Update learning rate scheduler"""
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

    def _check_improvement(
        self,
        val_loss: float,
        val_f1: float,
        val_fpr: float
    ) -> bool:
        """Check if model improved"""
        improved = False

        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            improved = True

        if val_f1 > self.state.best_val_f1:
            self.state.best_val_f1 = val_f1
            improved = True

        if val_fpr < self.state.best_val_fpr:
            self.state.best_val_fpr = val_fpr
            improved = True

        if improved:
            self.state.epochs_without_improvement = 0
        else:
            self.state.epochs_without_improvement += 1

        return improved

    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        return self.state.epochs_without_improvement >= self.early_stopping_patience

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_metrics: MetricResults,
        is_best: bool
    ):
        """Save checkpoint"""
        should_save = False

        if self.checkpoint_frequency == 'every_epoch':
            should_save = True
        elif self.checkpoint_frequency == 'every_5_epochs' and (epoch + 1) % 5 == 0:
            should_save = True
        elif self.checkpoint_frequency == 'every_10_epochs' and (epoch + 1) % 10 == 0:
            should_save = True
        elif self.checkpoint_frequency == 'best_only' and is_best:
            should_save = True

        if not should_save and not is_best:
            return

        # BUGFIX: Create checkpoint dict safely with error handling
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict(),
                'state': self.state,
                'config': self.config,
                'val_loss': val_loss,
                'val_metrics': val_metrics.to_dict()
            }
        except Exception as e:
            logger.error(f"Failed to create checkpoint dict: {e}")
            return

        # BUGFIX: Save with atomic write to prevent corruption
        if should_save:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            temp_path = checkpoint_path.with_suffix('.pt.tmp')
            try:
                torch.save(checkpoint, temp_path)
                temp_path.replace(checkpoint_path)  # Atomic rename
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                if temp_path.exists():
                    temp_path.unlink()  # Clean up temp file

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            temp_best_path = best_path.with_suffix('.pt.tmp')
            try:
                torch.save(checkpoint, temp_best_path)
                temp_best_path.replace(best_path)  # Atomic rename
                logger.info(f"Saved best model: {best_path}")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")
                if temp_best_path.exists():
                    temp_best_path.unlink()  # Clean up temp file

    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint with error handling"""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # BUGFIX: Validate checkpoint file exists and is readable
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint file: {e}")
            raise RuntimeError(f"Corrupted or invalid checkpoint file: {checkpoint_path}") from e
        
        # BUGFIX: Validate checkpoint structure
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'state']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.error(f"Failed to load model state dict: {e}")
            raise RuntimeError("Model state dict incompatible with current model") from e
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"Failed to load optimizer state dict: {e}. Continuing with fresh optimizer.")

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load scheduler state dict: {e}. Continuing with fresh scheduler.")
        
        if 'scaler_state_dict' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load scaler state dict: {e}. Continuing with fresh scaler.")
        
        self.state = checkpoint['state']

        logger.info(f"Checkpoint loaded: Epoch {self.state.epoch + 1}")

    def add_callback(self, callback: Callable):
        """Add training callback"""
        self.callbacks.append(callback)

    def _call_callbacks(self, event: str, *args, **kwargs):
        """Call all callbacks for event"""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(*args, **kwargs)

if __name__ == "__main__":
    # Test trainer initialization
    print("Trainer Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("⚠️  CUDA not available - trainer requires GPU")
        print("This is a basic initialization test only")

    # Create dummy model
    from src.models.architectures import create_model

    model = create_model('resnet18', num_classes=2, pretrained=False)
    print(f"✅ Created model: ResNet18")

    # Create dummy config
    from src.config.defaults import WakewordConfig

    config = WakewordConfig()
    print(f"✅ Created config")

    # Create dummy data loaders
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 1, 64, 50),  # Spectrograms
        torch.randint(0, 2, (100,))   # Labels
    )

    train_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=False)

    print(f"✅ Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Create trainer (will fail if CUDA not available due to enforce_cuda)
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_dir=Path("test_checkpoints"),
            device=device
        )
        print(f"✅ Trainer initialized successfully")

        print("\n✅ Trainer module loaded successfully")
        print("Note: Full training test requires actual dataset and GPU")

    except SystemExit as e:
        print("\n❌ Trainer requires CUDA GPU (as specified in requirements)")
        print("  This is expected behavior - CPU fallback not allowed")