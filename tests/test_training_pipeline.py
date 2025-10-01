"""
End-to-End Training Pipeline Test
Tests all components of the training pipeline
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.architectures import create_model
from src.models.losses import create_loss_function
from src.training.metrics import MetricsCalculator, MetricsTracker, calculate_class_weights
from src.training.optimizer_factory import create_optimizer, create_scheduler
from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager
from src.data.augmentation import AudioAugmentation, SpecAugment
from src.config.defaults import WakewordConfig


class TrainingPipelineTest:
    """Comprehensive training pipeline test suite"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_results = []
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)

        print("=" * 80)
        print("WAKEWORD TRAINING PIPELINE - END-TO-END TEST")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Test directory: {self.test_dir}")
        print("=" * 80)

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append((test_name, passed, details))
        print(f"{status}: {test_name}")
        if details:
            print(f"    {details}")

    def test_model_architectures(self):
        """Test all model architectures"""
        print("\n[1] Testing Model Architectures")
        print("-" * 80)

        architectures = ["resnet18", "mobilenetv3", "lstm", "gru", "tcn"]
        batch_size = 4

        for arch in architectures:
            try:
                model = create_model(arch, num_classes=2, pretrained=False)
                model = model.to(self.device)

                # Test forward pass
                if arch in ["resnet18", "mobilenetv3"]:
                    # 2D input (spectrograms)
                    test_input = torch.randn(batch_size, 1, 64, 50).to(self.device)
                else:
                    # Sequential input
                    test_input = torch.randn(batch_size, 50, 40).to(self.device)

                output = model(test_input)

                # Check output shape
                assert output.shape == (batch_size, 2), f"Unexpected output shape: {output.shape}"

                self.log_test(
                    f"Model Architecture: {arch}",
                    True,
                    f"Input: {test_input.shape}, Output: {output.shape}"
                )

            except Exception as e:
                self.log_test(f"Model Architecture: {arch}", False, str(e))

    def test_loss_functions(self):
        """Test loss functions"""
        print("\n[2] Testing Loss Functions")
        print("-" * 80)

        batch_size = 32
        num_classes = 2

        predictions = torch.randn(batch_size, num_classes).to(self.device)
        targets = torch.randint(0, num_classes, (batch_size,)).to(self.device)

        loss_functions = ["cross_entropy", "focal_loss"]

        for loss_name in loss_functions:
            try:
                criterion = create_loss_function(
                    loss_name=loss_name,
                    num_classes=num_classes,
                    label_smoothing=0.1,
                    device=self.device
                )

                loss = criterion(predictions, targets)

                # Check loss is scalar and finite
                assert loss.dim() == 0, "Loss should be scalar"
                assert torch.isfinite(loss), "Loss should be finite"

                self.log_test(
                    f"Loss Function: {loss_name}",
                    True,
                    f"Loss value: {loss.item():.4f}"
                )

            except Exception as e:
                self.log_test(f"Loss Function: {loss_name}", False, str(e))

    def test_metrics_calculation(self):
        """Test metrics calculation"""
        print("\n[3] Testing Metrics Calculation")
        print("-" * 80)

        batch_size = 100
        num_classes = 2

        predictions = torch.randn(batch_size, num_classes).to(self.device)
        targets = torch.randint(0, num_classes, (batch_size,)).to(self.device)

        try:
            calculator = MetricsCalculator(device=self.device)
            metrics = calculator.calculate(predictions, targets)

            # Check metrics are in valid ranges
            assert 0 <= metrics.accuracy <= 1, "Accuracy out of range"
            assert 0 <= metrics.precision <= 1, "Precision out of range"
            assert 0 <= metrics.recall <= 1, "Recall out of range"
            assert 0 <= metrics.f1_score <= 1, "F1 out of range"
            assert 0 <= metrics.fpr <= 1, "FPR out of range"
            assert 0 <= metrics.fnr <= 1, "FNR out of range"

            self.log_test(
                "Metrics Calculation",
                True,
                f"Acc={metrics.accuracy:.4f}, F1={metrics.f1_score:.4f}, FPR={metrics.fpr:.4f}"
            )

        except Exception as e:
            self.log_test("Metrics Calculation", False, str(e))

    def test_metrics_tracker(self):
        """Test metrics tracker"""
        print("\n[4] Testing Metrics Tracker")
        print("-" * 80)

        try:
            tracker = MetricsTracker(device=self.device)

            # Simulate 3 epochs
            for epoch in range(3):
                tracker.reset()

                # Simulate 5 batches per epoch
                for batch in range(5):
                    predictions = torch.randn(20, 2).to(self.device)
                    targets = torch.randint(0, 2, (20,)).to(self.device)
                    tracker.update(predictions, targets)

                # Compute epoch metrics
                metrics = tracker.compute()
                tracker.save_epoch_metrics(metrics)

            # Check history
            assert len(tracker.get_epoch_history()) == 3, "Should have 3 epochs"

            # Get best epoch
            best_epoch, best_metrics = tracker.get_best_epoch('f1_score')
            assert 0 <= best_epoch < 3, "Best epoch out of range"

            self.log_test(
                "Metrics Tracker",
                True,
                f"Tracked 3 epochs, Best F1 at epoch {best_epoch+1}: {best_metrics.f1_score:.4f}"
            )

        except Exception as e:
            self.log_test("Metrics Tracker", False, str(e))

    def test_class_weights(self):
        """Test class weights calculation"""
        print("\n[5] Testing Class Weights Calculation")
        print("-" * 80)

        dataset_stats = {'positive': 200, 'negative': 1800}  # 1:9 imbalance

        methods = ['balanced', 'inverse', 'sqrt_inverse']

        for method in methods:
            try:
                weights = calculate_class_weights(
                    dataset_stats,
                    method=method,
                    device=self.device
                )

                # Check weights shape
                assert weights.shape == (2,), "Weights should be 2D"

                # Check weights are positive
                assert (weights > 0).all(), "Weights should be positive"

                self.log_test(
                    f"Class Weights: {method}",
                    True,
                    f"Neg={weights[0].item():.4f}, Pos={weights[1].item():.4f}"
                )

            except Exception as e:
                self.log_test(f"Class Weights: {method}", False, str(e))

    def test_augmentation(self):
        """Test data augmentation"""
        print("\n[6] Testing Data Augmentation")
        print("-" * 80)

        try:
            # Test audio augmentation
            augmentation = AudioAugmentation(
                sample_rate=16000,
                device=self.device,
                time_stretch_range=(0.8, 1.2),
                pitch_shift_range=(-2, 2),
                background_noise_prob=0.5
            )

            # Test on dummy audio
            test_audio = torch.randn(1, 16000).to(self.device)
            augmented = augmentation(test_audio)

            assert augmented.shape == test_audio.shape, "Augmented shape mismatch"
            assert torch.isfinite(augmented).all(), "Augmented audio has NaN/Inf"

            self.log_test(
                "Audio Augmentation",
                True,
                f"Input: {test_audio.shape}, Output: {augmented.shape}"
            )

            # Test SpecAugment
            spec_aug = SpecAugment(
                freq_mask_param=15,
                time_mask_param=35,
                n_freq_masks=2,
                n_time_masks=2
            )

            test_spec = torch.randn(1, 64, 50).to(self.device)
            augmented_spec = spec_aug(test_spec)

            assert augmented_spec.shape == test_spec.shape, "SpecAugment shape mismatch"

            self.log_test(
                "SpecAugment",
                True,
                f"Input: {test_spec.shape}, Output: {augmented_spec.shape}"
            )

        except Exception as e:
            self.log_test("Data Augmentation", False, str(e))

    def test_optimizer_and_scheduler(self):
        """Test optimizer and scheduler creation"""
        print("\n[7] Testing Optimizer and Scheduler")
        print("-" * 80)

        # Create dummy model
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        ).to(self.device)

        optimizers = ['adam', 'adamw', 'sgd']
        schedulers = ['cosine', 'step', 'plateau', 'none']

        for opt_name in optimizers:
            try:
                optimizer = create_optimizer(
                    model,
                    optimizer_name=opt_name,
                    learning_rate=0.001,
                    weight_decay=1e-4
                )

                # Test optimizer step
                loss = torch.randn(1, requires_grad=True).to(self.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.log_test(
                    f"Optimizer: {opt_name}",
                    True,
                    f"Type: {type(optimizer).__name__}"
                )

            except Exception as e:
                self.log_test(f"Optimizer: {opt_name}", False, str(e))

        for sched_name in schedulers:
            try:
                optimizer = create_optimizer(model, optimizer_name='adam', learning_rate=0.001)
                scheduler = create_scheduler(
                    optimizer,
                    scheduler_name=sched_name,
                    epochs=50,
                    warmup_epochs=0
                )

                # Test scheduler step
                if scheduler is not None:
                    if sched_name == 'plateau':
                        scheduler.step(0.5)
                    else:
                        scheduler.step()

                self.log_test(
                    f"Scheduler: {sched_name}",
                    True,
                    f"Type: {type(scheduler).__name__ if scheduler else 'None'}"
                )

            except Exception as e:
                self.log_test(f"Scheduler: {sched_name}", False, str(e))

    def test_training_loop(self):
        """Test complete training loop"""
        print("\n[8] Testing Training Loop (Mini Training)")
        print("-" * 80)

        if self.device == "cpu":
            self.log_test(
                "Training Loop",
                False,
                "CUDA required for training (as specified in requirements)"
            )
            return

        try:
            # Create configuration
            config = WakewordConfig()
            config.training.epochs = 3  # Quick test
            config.training.batch_size = 8
            config.training.early_stopping_patience = 10

            # Create model
            model = create_model('resnet18', num_classes=2, pretrained=False)

            # Create dummy dataset
            num_samples = 100
            dummy_features = torch.randn(num_samples, 1, 64, 50)
            dummy_labels = torch.randint(0, 2, (num_samples,))

            dataset = TensorDataset(dummy_features, dummy_labels)
            train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

            # Create trainer
            checkpoint_dir = self.test_dir / "checkpoints"
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                checkpoint_dir=checkpoint_dir,
                device=self.device
            )

            # Train for 3 epochs
            results = trainer.train()

            # Check results
            assert 'history' in results, "Missing history in results"
            assert len(results['history']['train_loss']) == 3, "Should have 3 epochs"
            assert results['final_epoch'] == 2, "Final epoch should be 2"

            self.log_test(
                "Training Loop",
                True,
                f"Trained for 3 epochs, Final val loss: {results['best_val_loss']:.4f}"
            )

        except Exception as e:
            self.log_test("Training Loop", False, str(e))

    def test_checkpointing(self):
        """Test checkpoint management"""
        print("\n[9] Testing Checkpoint Management")
        print("-" * 80)

        try:
            checkpoint_dir = self.test_dir / "checkpoints"

            if not checkpoint_dir.exists():
                self.log_test(
                    "Checkpoint Management",
                    False,
                    "No checkpoints created (previous test may have failed)"
                )
                return

            # Create checkpoint manager
            manager = CheckpointManager(checkpoint_dir)

            # List checkpoints
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) > 0, "No checkpoints found"

            # Get best checkpoint
            best = manager.get_best_checkpoint(metric='val_f1', mode='max')
            assert best is not None, "Best checkpoint not found"

            self.log_test(
                "Checkpoint Management",
                True,
                f"Found {len(checkpoints)} checkpoints, Best at epoch {best.epoch+1}"
            )

        except Exception as e:
            self.log_test("Checkpoint Management", False, str(e))

    def test_checkpoint_loading(self):
        """Test checkpoint loading and resuming"""
        print("\n[10] Testing Checkpoint Loading")
        print("-" * 80)

        if self.device == "cpu":
            self.log_test(
                "Checkpoint Loading",
                False,
                "CUDA required for this test"
            )
            return

        try:
            checkpoint_dir = self.test_dir / "checkpoints"

            if not checkpoint_dir.exists():
                self.log_test(
                    "Checkpoint Loading",
                    False,
                    "No checkpoints available (previous test may have failed)"
                )
                return

            # Create model
            model = create_model('resnet18', num_classes=2, pretrained=False)

            # Load checkpoint
            manager = CheckpointManager(checkpoint_dir)
            checkpoint = manager.load_best_model(model, device=self.device)

            assert checkpoint is not None, "Failed to load checkpoint"
            assert 'epoch' in checkpoint, "Missing epoch in checkpoint"

            self.log_test(
                "Checkpoint Loading",
                True,
                f"Loaded checkpoint from epoch {checkpoint['epoch']+1}"
            )

        except Exception as e:
            self.log_test("Checkpoint Loading", False, str(e))

    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 80)
        print("TEST REPORT")
        print("=" * 80)

        passed = sum(1 for _, p, _ in self.test_results if p)
        failed = len(self.test_results) - passed

        print(f"\nTotal tests: {len(self.test_results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success rate: {passed / len(self.test_results) * 100:.1f}%")

        if failed > 0:
            print("\nFailed tests:")
            for name, passed, details in self.test_results:
                if not passed:
                    print(f"  ❌ {name}")
                    if details:
                        print(f"     {details}")

        print("\n" + "=" * 80)

        if failed == 0:
            print("✅ ALL TESTS PASSED - TRAINING PIPELINE READY")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED - REVIEW REQUIRED")

        print("=" * 80)

    def cleanup(self):
        """Clean up test files"""
        print("\nCleaning up test files...")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"✅ Removed test directory: {self.test_dir}")

    def run_all_tests(self):
        """Run all tests"""
        self.test_model_architectures()
        self.test_loss_functions()
        self.test_metrics_calculation()
        self.test_metrics_tracker()
        self.test_class_weights()
        self.test_augmentation()
        self.test_optimizer_and_scheduler()
        self.test_training_loop()
        self.test_checkpointing()
        self.test_checkpoint_loading()

        self.generate_report()
        self.cleanup()


if __name__ == "__main__":
    # Run end-to-end test
    test_suite = TrainingPipelineTest()
    test_suite.run_all_tests()