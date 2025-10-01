# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Wakeword Training Platform** - a GPU-accelerated, Gradio-based application for training custom wakeword detection models. The project is designed to handle large-scale datasets (300k+ audio files) with a strict GPU-only policy (no CPU fallback for training operations).

**Key Philosophy**: Reliability over creativity. The project follows industry-standard practices and enforces CUDA availability for all tensor operations.

## Common Commands

### Launch the Application
```bash
python src/ui/app.py
# or
python -m src.ui.app
```
The UI auto-selects an available port between 7860-7870 and opens at `http://localhost:XXXX`.

### Verify Installation
```bash
python verify_installation.py
```
Checks CUDA availability, GPU status, and all dependencies.

### Run Tests
```bash
pytest tests/
# Run specific test
pytest tests/test_training_pipeline.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
**Note**: Uses PyTorch 2.1.2 with CUDA 11.8. Versions are pinned for compatibility.

## Architecture Overview

### Core Design Principles

1. **Mandatory GPU Acceleration**: No CPU fallback for training/inference. All tensor operations require CUDA.
2. **6-Panel Gradio Interface**: Sequential workflow from dataset → config → training → evaluation → export → docs.
3. **Configuration-Driven**: Dataclass-based configs (`src/config/defaults.py`) with presets for different use cases.
4. **Modular Pipeline**: Each component (data, models, training, evaluation) is independent and testable.

### Application Flow

```
Panel 1 (Dataset)  →  Panel 2 (Config)  →  Panel 3 (Training)
       ↓                      ↓                     ↓
   Scan/Split          Configure Params      Train Models
   Extract .npy        Save/Load Configs     Live Metrics
       ↓                      ↓                     ↓
Panel 4 (Evaluation) ← Panel 5 (Export)  ←  Panel 6 (Docs)
       ↓                      ↓
   Test Models           ONNX Export          Knowledge Base
   Microphone Test       Quantization         Troubleshooting
```

### Key Architectural Patterns

#### 1. CUDA Enforcement
All training/inference code enforces CUDA via `src/config/cuda_utils.py`:
```python
from src.config.cuda_utils import enforce_cuda
cuda_validator = enforce_cuda()  # Raises error if CUDA unavailable
```

#### 2. Configuration Management
Uses dataclass hierarchy (`WakewordConfig` → `DataConfig`, `TrainingConfig`, etc.):
- Save/load as YAML
- Validation in `src/config/validator.py`
- Presets in `src/config/presets.py`

#### 3. Dataset Pipeline
```
Raw Audio Files → AudioProcessor → WakewordDataset → DataLoader → Training
                       ↓
                  Augmentation (GPU-accelerated)
                  - Background noise mixing
                  - RIR convolution
                  - Time/pitch shifting
```

**Expected Dataset Structure**:
```
data/raw/
├── positive/           # Wakeword utterances
├── negative/           # Non-wakeword speech
├── hard_negative/      # Similar-sounding phrases
├── background/         # Environmental noise
├── rirs/              # Room impulse responses
└── npy/               # Pre-computed features (optional)
```

#### 4. Model Factory Pattern
`src/models/architectures.py::create_model()` returns models based on architecture name:
- `"resnet18"` → ResNet18Wakeword (recommended for accuracy)
- `"mobilenetv3"` → MobileNetV3Wakeword (edge deployment)
- `"lstm"`, `"gru"`, `"tcn"` → Sequential models

All models output `(batch_size, num_classes)` logits.

#### 5. Training Loop
`src/training/trainer.py::Trainer` handles:
- Mixed precision training (AMP)
- Gradient clipping
- Checkpoint management (best/last models based on `checkpoint_frequency`)
- Early stopping
- Metrics tracking (accuracy, FPR, FNR, F1)

**Critical**: Training uses `enforce_cuda()` and will fail gracefully if GPU unavailable.

## Important Implementation Details

### GPU Memory Management
- **Batch Size Auto-Adjustment**: Not implemented yet; reduce manually if OOM occurs.
- **Mixed Precision**: Enabled by default (`config.optimizer.mixed_precision = True`).
- **Pin Memory**: Always enabled in DataLoaders for faster GPU transfer.
- **Num Workers**: Default is 4; adjust based on CPU cores (formula in plan: `max(cpu_count - 8, 8)`).

### Audio Processing
- **Format**: All audio converted to 16kHz, 16-bit, mono on load.
- **Duration**: Fixed-length chunks (default 1.5s). Pad/trim as needed.
- **Features**: Mel-spectrogram (default) or MFCC, computed on-the-fly on GPU.
- **Augmentation**: Applied during training via `src/data/augmentation.py` with torchaudio transforms.

### Dataset Splitting
- **Ratios**: Train 70-80%, Val 10-15%, Test 10-15% (configurable in Panel 1).
- **Strategy**: Stratified splitting to maintain class balance.
- **Manifests**: Saved as JSON in `data/splits/` with file paths and labels.

### Checkpoint Management
`checkpoint_frequency` options (in `TrainingConfig`):
- `"best_only"`: Save only when validation loss improves
- `"every_epoch"`: Save after every epoch
- `"every_5_epochs"`, `"every_10_epochs"`: Periodic saves

Checkpoints saved to `models/checkpoints/` with format: `{model_name}_epoch_{N}.pth`.

### Metrics Priority
1. **False Positive Rate (FPR)**: Critical for user experience (target <5%, ideally <2%)
2. **False Negative Rate (FNR)**: Important but less critical (target <5%)
3. **Accuracy**: Overall performance (target >95%)
4. **F1-Score**: Balance between precision/recall

Tracked in real-time via `src/training/metrics.py::MetricsTracker`.

## Dataset Requirements (Industry Standards)

From the implementation plan and README:
- **Minimum Positive Samples**: 1,000 (5,000+ recommended)
- **Negative:Positive Ratio**: 8:1 to 10:1
- **Hard Negatives**: 20-30% of total negatives
- **Speaker Diversity**: 50+ unique voices
- **Audio Quality**: 16kHz, 16-bit, mono, 1.5-2 seconds duration

## Configuration Presets

Defined in `src/config/presets.py`:
- `Default`: Balanced for general use
- `Small Dataset`: Aggressive augmentation for <10k samples
- `Large Dataset`: Optimized for >100k samples
- `Fast Training`: Minimal augmentation for quick iteration
- `High Accuracy`: Extended training, conservative augmentation
- `Edge Deployment`: Model size optimization for mobile/IoT

## ONNX Export

Panel 5 exports trained models to ONNX format via `src/export/onnx_exporter.py`:
- **Opset**: Default 14 (configurable)
- **Quantization**: FP16 or INT8 options
- **Dynamic Batch Size**: Enabled by default
- **Validation**: Compares PyTorch vs ONNX outputs

## Logging

All modules use Python logging:
- Logs saved to `logs/` directory with timestamps
- Per-module loggers: `data_YYYYMMDD_HHMMSS.log`, `config_*.log`, `app_*.log`
- Logger setup in `src/config/logger.py`

## Development Guidelines

### When Adding New Features

1. **Always enforce CUDA**: Import and call `enforce_cuda()` in training/inference code.
2. **Use dataclass configs**: Extend existing config classes in `src/config/defaults.py`.
3. **Update presets**: If adding new config parameters, update all presets in `src/config/presets.py`.
4. **Add validation**: Update `src/config/validator.py` for new parameters.
5. **Maintain GPU-only policy**: Do NOT add CPU fallback for tensor operations.

### When Modifying Models

- All models must return `(batch_size, num_classes)` logits.
- Register new architectures in `create_model()` factory function.
- Follow existing patterns: ResNet/MobileNet for 2D (spectrograms), LSTM/GRU/TCN for sequential data.
- Test with both 2D and sequential inputs as appropriate.

### When Working with Audio

- Always use `AudioProcessor` from `src/data/audio_utils.py` for loading/preprocessing.
- Augmentations in `src/data/augmentation.py` are GPU-accelerated; keep them on CUDA.
- Handle corrupted files gracefully (log error, skip file, continue).

### Port Selection

The app auto-finds available ports in range 7860-7870. If all busy, it fails with clear error. Implemented in `src/ui/app.py::find_available_port()`.

## Troubleshooting Common Issues

### CUDA Not Available
Run `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`. Reinstall PyTorch with CUDA 11.8 if needed.

### Out of Memory
- Reduce `batch_size` in configuration
- Use shorter `audio_duration`
- Choose lighter model (`mobilenetv3`)
- Enable mixed precision (should be on by default)

### Training Not Starting
- Check dataset splits exist in `data/splits/`
- Verify manifest JSON files are valid
- Check logs in `logs/` directory
- Ensure GPU has enough memory for batch size

### Slow Data Loading
- Increase `num_workers` in TrainingConfig (but not beyond CPU cores)
- Enable `persistent_workers=True` (default)
- Check if audio files are on slow storage (network drive)

## Sprint Status

Project completed through **Sprint 7**:
- ✅ Sprint 1: Foundation (structure, CUDA validation, logging, 6-panel skeleton)
- ✅ Sprint 2: Dataset Management (scanner, splitter, .npy extractor, health checker)
- ✅ Sprint 3: Configuration (defaults, validator, presets, save/load)
- ✅ Sprint 4: Training Pipeline (Dataset, augmentation, models, trainer, metrics)
- ✅ Sprint 5: Training UI (Panel 3 with live metrics, plots, controls)
- ✅ Sprint 6: Evaluation (file-based, microphone, test set evaluation)
- ✅ Sprint 7: Export & Docs (ONNX exporter, Panel 5/6)

See `docs/SPRINTX_COMPLETE.md` for detailed sprint summaries.

## File Organization Logic

- `src/config/`: Configuration classes, validation, CUDA utils, logging
- `src/data/`: Dataset classes, augmentation, audio processing, health checking
- `src/models/`: Model architectures and loss functions
- `src/training/`: Training loop, metrics tracking, optimizer factory, checkpoint manager
- `src/evaluation/`: Evaluation tools, inference engine
- `src/export/`: ONNX conversion
- `src/ui/`: Gradio panels (one file per panel) and main app

## Critical Constraints

1. **No CPU Fallback**: Training/inference MUST use CUDA. Fail fast with clear error if GPU unavailable.
2. **No Empty Commits**: Only create git commits when explicitly requested.
3. **Reliability First**: Prefer proven patterns over experimental approaches.
4. **Version Pinning**: Don't upgrade package versions without testing compatibility.
5. **Industry Standards**: Follow best practices from wakeword detection research (see Implementation_plan.md Appendix).

## Additional Resources

- **Implementation Plan**: `Implementation_plan.md` - comprehensive technical specifications
- **README**: `README.md` - user-facing documentation and quick start
- **Sprint Docs**: `docs/SPRINTX_COMPLETE.md` - detailed completion summaries
- **In-App Documentation**: Panel 6 contains knowledge base with dataset prep, config explanations, troubleshooting
