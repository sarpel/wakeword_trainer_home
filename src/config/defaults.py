"""
Default Configuration Parameters for Wakeword Training
Defines basic and advanced training hyperparameters
"""
from typing import Dict, Any, List
from dataclasses import dataclass, asdict, field
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Audio parameters
    sample_rate: int = 16000  # Hz
    audio_duration: float = 2.5  # seconds
    n_mfcc: int = 40  # Number of MFCC coefficients
    n_fft: int = 1024  # FFT window size
    hop_length: int = 160  # Hop length for STFT
    n_mels: int = 128  # Number of mel bands

    # Feature extraction
    feature_type: str = "mel"  # mel, mfcc
    normalize_audio: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

    # Hardware
    num_workers: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True

    # Checkpointing
    checkpoint_frequency: str = "best_only"  # best_only, every_epoch, every_5_epochs, every_10_epochs
    save_best_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str = "resnet18"  # resnet18, mobilenetv3, lstm, gru, tcn
    num_classes: int = 2  # Binary classification
    pretrained: bool = True  # Use pretrained weights (ImageNet)
    dropout: float = 0.3

    # Architecture-specific parameters
    hidden_size: int = 128  # For LSTM/GRU
    num_layers: int = 2  # For LSTM/GRU
    bidirectional: bool = True  # For LSTM/GRU

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    # Time domain augmentation
    time_stretch_min: float = 0.80
    time_stretch_max: float = 1.20
    pitch_shift_min: int = -2  # semitones
    pitch_shift_max: int = 2  # semitones

    # Noise augmentation
    background_noise_prob: float = 0.5
    noise_snr_min: float = 5.0  # dB
    noise_snr_max: float = 20.0  # dB

    # RIR augmentation
    rir_prob: float = 0.25

    # Frequency domain augmentation
    freq_mask_prob: float = 0.5
    time_mask_prob: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler configuration"""
    optimizer: str = "adamw"  # adam, sgd, adamw
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    # Scheduler
    scheduler: str = "cosine"  # cosine, step, plateau, none
    warmup_epochs: int = 3
    min_lr: float = 3e-4

    # Step scheduler parameters
    step_size: int = 10
    gamma: float = 0.1

    # Plateau scheduler parameters
    patience: int = 15
    factor: float = 0.5

    # Gradient
    gradient_clip: float = 1.0
    mixed_precision: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LossConfig:
    """Loss function configuration"""
    loss_function: str = "cross_entropy"  # cross_entropy, focal_loss
    label_smoothing: float = 0.05

    # Focal loss parameters
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Class weighting
    class_weights: str = "balanced"  # balanced, none, custom
    hard_negative_weight: float = 2.5

    # Sampling strategy
    sampler_strategy: str = "weighted"  # weighted, balanced, none

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class WakewordConfig:
    """Complete wakeword training configuration"""
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # Metadata
    config_name: str = "default"
    description: str = "Default wakeword training configuration"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'config_name': self.config_name,
            'description': self.description,
            'data': self.data.to_dict(),
            'training': self.training.to_dict(),
            'model': self.model.to_dict(),
            'augmentation': self.augmentation.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'loss': self.loss.to_dict()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WakewordConfig':
        """Create configuration from dictionary"""
        return cls(
            config_name=config_dict.get('config_name', 'default'),
            description=config_dict.get('description', ''),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            augmentation=AugmentationConfig(**config_dict.get('augmentation', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            loss=LossConfig(**config_dict.get('loss', {}))
        )

    def save(self, path: Path):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)  # <-- safe_dump

    @classmethod
    def load(cls, path: Path) -> 'WakewordConfig':
        """Load configuration from YAML file"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)


def get_default_config() -> WakewordConfig:
    """
    Get default configuration

    Returns:
        Default WakewordConfig instance
    """
    return WakewordConfig(
        config_name="default",
        description="Default balanced configuration for general use"
    )


# Export all configurations
__all__ = [
    'DataConfig',
    'TrainingConfig',
    'ModelConfig',
    'AugmentationConfig',
    'OptimizerConfig',
    'LossConfig',
    'WakewordConfig',
    'get_default_config'
]


if __name__ == "__main__":
    # Test configuration system
    print("Configuration System Test")
    print("=" * 60)

    # Create default config
    config = get_default_config()
    print("\nDefault Configuration:")
    print(f"  Sample Rate: {config.data.sample_rate} Hz")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Model: {config.model.architecture}")

    # Test save/load
    test_path = Path("test_config.yaml")
    config.save(test_path)
    print(f"\n✅ Configuration saved to: {test_path}")

    # Load back
    loaded_config = WakewordConfig.load(test_path)
    print(f"✅ Configuration loaded successfully")
    print(f"  Loaded config name: {loaded_config.config_name}")

    # Cleanup
    test_path.unlink()
    print(f"✅ Test file cleaned up")

    print("\nConfiguration system test complete")