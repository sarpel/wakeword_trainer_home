# Wakeword Training Platform - Implementation Plan

## Project Overview
A Gradio-based wakeword training platform with GPU/CUDA acceleration for processing 300k+ audio files with industry-standard practices.

## Core Requirements Summary
- **Scale**: 300k audio files processing
- **Hardware**: GPU acceleration with CUDA and PyTorch
- **Dataset Types**: Positive, Negative, Hard-negative, Background noise, RIRs, .npy files
- **Architecture**: 6-panel Gradio interface
- **Focus**: Reliability over creativity, industry-standard practices

---

## Technology Stack

### Core Dependencies
- **Python**: 3.8+ (CUDA compatibility)
- **UI Framework**: Gradio (latest stable)
- **ML Framework**: PyTorch with CUDA support
- **Audio Processing**: librosa, soundfile, scipy
- **Data Handling**: numpy, pandas
- **Model Export**: onnx, onnxruntime
- **Visualization**: matplotlib, plotly
- **Audio Recording**: sounddevice, pyaudio

### Development Environment
- CUDA Toolkit (11.8+ or 12.x)
- cuDNN libraries
- Virtual environment with pinned dependencies

---

## Implementation Phases

## Phase 1: Project Structure & Environment Setup

### 1.1 Directory Structure
```
project_root/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── defaults.py          # Default hyperparameters
│   │   └── validator.py         # Config validation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   ├── augmentation.py      # Audio augmentation
│   │   ├── splitter.py          # Train/test/val splitting
│   │   └── npy_extractor.py     # .npy file processing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures.py     # Model definitions
│   │   └── losses.py            # Loss functions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── callbacks.py         # Training callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py         # Model evaluation
│   │   └── inference.py         # Real-time inference
│   ├── export/
│   │   ├── __init__.py
│   │   └── onnx_converter.py    # ONNX export
│   └── ui/
│       ├── __init__.py
│       ├── app.py               # Main Gradio app
│       ├── panel_dataset.py     # Panel 1
│       ├── panel_config.py      # Panel 2
│       ├── panel_training.py    # Panel 3
│       ├── panel_evaluation.py  # Panel 4
│       ├── panel_export.py      # Panel 5
│       └── panel_docs.py        # Panel 6
├── data/
│   ├── raw/
│   │   ├── positive/
│   │   ├── negative/
│   │   ├── hard_negative/
│   │   ├── background/
│   │   ├── rirs/
│   │   └── npy/
│   ├── processed/
│   │   ├── train/
│   │   ├── test/
│   │   └── validate/
│   └── splits/
│       └── split_manifest.json
├── models/
│   ├── checkpoints/
│   └── exported/
├── logs/
│   ├── training/
│   └── evaluation/
├── docs/
│   └── knowledge_base.md
├── requirements.txt
├── setup.py
└── README.md
```

### 1.2 Environment Setup Tasks
- Create requirements.txt with pinned versions
- Go for the most compatible package versions known in forums. It doesnt matter old or new, it just should work together.
- Setup CUDA detection and validation
- Create installation verification script
- Setup logging infrastructure

---

## Phase 2: Panel 1 - Dataset Management

### 2.1 Dataset Discovery & Validation
**File**: `src/data/splitter.py`

**Features**:
- Recursive scan of top-level folders (positive, negative, hard_negative, background, rirs)
- Include subfolders of the top folders and search audio files in there too. They all will not be ready in one single folder when users put their datasets in there.
- Audio format validation (wav, mp3, flac, ogg)
- File integrity checks (corrupted audio detection)
- Dataset statistics generation (duration, sample rate, channels)
- GPU-accelerated audio loading validation

**Output**:
- JSON manifest with file paths, labels, metadata
- Statistics report (total files, total duration per category)

### 2.2 Train/Test/Validation Splitting
**File**: `src/data/splitter.py`

**Industry Standard Ratios**:
- **Training**: 70-80%
- **Validation**: 10-15%
- **Testing**: 10-15%

**Split Strategy**:
- Stratified splitting (maintain class balance)
- Speaker-independent splitting (if metadata available)
- Temporal splitting for sequential data
- Configurable ratios with validation

**Implementation**:
- sklearn.model_selection.train_test_split
- Save split manifests (train.json, val.json, test.json)
- Ensure no data leakage between splits
- Handle imbalanced datasets with warnings

### 2.3 .npy File Extraction
**File**: `src/data/npy_extractor.py`

**Features**:
- Load .npy files containing pre-computed features
- Validate shape and dtype
- Convert to appropriate format for training
- Support both raw audio and feature arrays
- GPU memory-mapped loading for large files

**Supported .npy Formats**:
- Raw waveforms: (N, samples)
- Spectrograms: (N, freq_bins, time_steps)
- MFCC features: (N, n_mfcc, time_steps)

### 2.4 Dataset Health Briefing
**File**: `src/ui/panel_dataset.py`

**Pre-Training Validation Report**:
- Class distribution (positive vs negative ratio)
  - **Industry Standard**: 1:8 to 1:10 (positive:negative)
- Duration statistics per class
- Sample rate consistency check
- Audio quality metrics (SNR estimates)
- Recommended augmentation strategies
- Warning flags for problematic distributions

**UI Components**:
- File browser for dataset root selection
- "Scan Datasets" button
- Statistics display (tables + charts)
- "Split Datasets" button with ratio sliders
- "Extract .npy Files" button
- Health report panel with color-coded warnings

---

## Phase 3: Panel 2 - Configuration Management

### 3.1 Configuration Categories
**File**: `src/config/defaults.py`

#### Basic Parameters
```python
BASIC_CONFIG = {
    # Data
    'sample_rate': 16000,
    'audio_duration': 1.0,  # seconds
    'n_mfcc': 40,
    'n_fft': 512,
    'hop_length': 160,

    # Training
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,

    # Model
    'model_architecture': 'resnet18',  # Options: resnet18, mobilenet, lstm, tcn
    'num_classes': 2,  # binary classification
}
```

#### Advanced Parameters
```python
ADVANCED_CONFIG = {
    # Augmentation
    'time_stretch_rate': (0.8, 1.2),
    'pitch_shift_steps': (-2, 2),
    'background_noise_prob': 0.5,
    'noise_snr_db': (5, 20),
    'rir_prob': 0.3,

    # Training
    'optimizer': 'adam',  # adam, sgd, adamw
    'scheduler': 'cosine',  # cosine, step, plateau
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    'mixed_precision': True,
    'num_workers': 4,

    # Loss
    'loss_function': 'cross_entropy',  # cross_entropy, focal_loss
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2.0,
    'label_smoothing': 0.1,

    # Data Sampling
    'class_weights': 'balanced',  # balanced, none, custom
    'hard_negative_weight': 2.0,
    'sampler_strategy': 'weighted',  # weighted, balanced
}
```

### 3.2 Configuration UI
**File**: `src/ui/panel_config.py`

**UI Components**:
- Tabbed interface: "Basic" and "Advanced"
- Interactive sliders for numeric values
- Dropdowns for categorical choices
- Tooltips with parameter explanations
- "Load Preset" dropdown (Small Dataset, Large Dataset, Fast Training, High Accuracy)
- "Save Configuration" button (JSON export)
- "Load Configuration" button (JSON import)
- "Reset to Defaults" button
- Real-time validation with error messages

### 3.3 Configuration Validation
**File**: `src/config/validator.py`

**Validation Rules**:
- sample_rate compatibility with CUDA
- batch_size must fit in GPU memory
- audio_duration > 0
- learning_rate in reasonable range
- Cross-parameter validation (e.g., n_fft vs hop_length)
- GPU memory estimation based on config

---

## Phase 4: Panel 3 - Model Training

### 4.1 Dataset Loading
**File**: `src/data/dataset.py`

**PyTorch Dataset Class**:
- Custom `WakewordDataset(torch.utils.data.Dataset)`
- GPU-accelerated audio loading with torchaudio
- On-the-fly augmentation pipeline
- Background noise mixing on GPU
- RIR convolution on GPU
- Memory-efficient data streaming for 300k files
- Pre-loading strategy with pinned memory

**DataLoader Configuration**:
- `num_workers` based on CPU cores
- `pin_memory=True` for GPU transfer
- `persistent_workers=True` for efficiency
- Custom collate function for variable-length audio

### 4.2 Model Architectures
**File**: `src/models/architectures.py`

**Supported Architectures** (Industry-Proven):
1. **ResNet-18** (Recommended for accuracy)
   - Spectrogram input (mel or MFCC)
   - Modified first conv for audio
   - Binary classification head

2. **MobileNetV3-Small** (Recommended for edge deployment)
   - Lightweight, fast inference
   - Good accuracy-speed tradeoff

3. **LSTM/GRU** (Sequential processing)
   - Raw waveform or features
   - Bidirectional option

4. **TCN (Temporal Convolutional Network)** (Modern alternative)
   - Parallel training advantage
   - Long-range dependencies

**Implementation**:
- Use torchvision.models where applicable
- Custom modifications for audio input
- Pre-trained weights support (ImageNet transfer learning)

### 4.3 Training Loop
**File**: `src/training/trainer.py`

**Features**:
- CUDA device management with automatic GPU detection
- Mixed precision training (torch.cuda.amp)
- Gradient accumulation for large batch simulation
- Model checkpointing (best model, last model)(Configurable for their frequency)
- TensorBoard logging
- Learning rate scheduling
- Early stopping
- Graceful interruption handling (save state)

**Training Metrics**:
- Loss (train, validation)
- Accuracy (train, validation)
- Precision, Recall, F1-score
- False Positive Rate (FPR) - Critical for wakeword
- False Negative Rate (FNR)
- Epoch time, Current/Total Batch progress, samples/second

### 4.4 Live Training Visualization
**File**: `src/ui/panel_training.py`

**UI Components**:
- "Start Training" button
- "Pause Training" button
- "Stop Training" button
- Real-time metrics display (updated every batch/epoch)
- Live plot with dual y-axes (loss and accuracy)
- Progress bar for epochs and batches
- ETA estimation
- GPU utilization monitor
- Console log output

**Plotting Strategy**:
- Gradio Plot component updated via generator
- matplotlib backend for figure generation
- Separate plots: Loss, Accuracy, Learning Rate
- Both train and validation curves

**Text Metrics Display**:
```
Epoch: 15/50 | Batch: 450/1200
Train Loss: 0.1234 | Train Acc: 95.67%
Val Loss: 0.1567 | Val Acc: 94.32%
FPR: 2.1% | FNR: 3.5%
Speed: 342 samples/sec | GPU: 87%
ETA: 2h 15m
```

### 4.5 Multi-GPU Support (Future Enhancement)
- torch.nn.DataParallel for single-node multi-GPU
- Automatic batch size scaling
- Configuration flag for GPU selection

---

## Phase 5: Panel 4 - Model Evaluation

### 5.1 File-Based Evaluation
**File**: `src/evaluation/evaluator.py`

**Features**:
- Upload single or multiple audio files
- Batch evaluation with progress bar
- GPU-accelerated inference
- Confidence score output (0-1)
- Threshold adjustment slider
- Confusion matrix visualization
- ROC curve (if ground truth provided)
- Detection latency measurement

**UI Components**:
- File upload (drag-and-drop, multi-select)
- "Evaluate Files" button
- Results table (filename, prediction, confidence, latency)
- Threshold slider (default: 0.5)
- Export results as CSV

### 5.2 Real-Time Microphone Testing
**File**: `src/evaluation/inference.py`

**Features**:
- Live audio capture from microphone
- Streaming inference (sliding window)
- Configurable buffer size and hop
- Visual waveform display
- Detection trigger with confidence
- Audio playback of captured segments

**Implementation**:
- sounddevice for audio capture
- Ring buffer for continuous processing
- GPU inference with batched windows
- Post-processing (smoothing, debouncing)

**UI Components**:
- Microphone selector dropdown
- "Start Recording" / "Stop Recording" toggle
- Live waveform plot
- Detection indicator (green light + confidence)
- Detection history log
- Sensitivity slider

### 5.3 Test Set Evaluation
**Features**:
- Automatic evaluation on test split
- Comprehensive metrics report
- Per-class performance breakdown
- Error analysis (false positives/negatives)
- Optimal threshold recommendation

---

## Phase 6: Panel 5 - ONNX Export

### 6.1 Model Conversion
**File**: `src/export/onnx_converter.py`

**Features**:
- Load PyTorch checkpoint
- Convert to ONNX format
- Opset version selection (default: 14)
- Dynamic batch size support
- Quantization options (FP16, INT8)
- ONNX model validation
- Inference speed comparison (PyTorch vs ONNX)

**Implementation**:
```python
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['audio'],
    output_names=['logits'],
    dynamic_axes={'audio': {0: 'batch_size'}}
)
```

**UI Components**:
- Checkpoint selector dropdown
- Opset version number input
- "Export to ONNX" button
- Quantization checkbox (FP16, INT8)
- Export progress indicator
- Validation results display
- Download exported model button

### 6.2 ONNX Testing
**Features**:
- Load and test ONNX model
- Compare outputs with PyTorch model
- Inference speed benchmark
- ONNX Runtime compatibility check

---

## Phase 7: Panel 6 - Documentation & Knowledge Base

### 7.1 Documentation Sections
**File**: `src/ui/panel_docs.py` + `docs/knowledge_base.md`

**Content Structure**:

#### 1. Introduction
- What is a wakeword?
- Use cases and applications
- System requirements

#### 2. Dataset Preparation
- Audio file formats and requirements
- Dataset types explained:
  - **Positive**: Actual wakeword utterances
  - **Negative**: Non-wakeword speech
  - **Hard Negative**: Similar-sounding phrases
  - **Background Noise**: Environmental sounds
  - **RIRs**: Room Impulse Responses for reverberation
- Dataset size recommendations
- Recording best practices
- Data quality guidelines

#### 3. Dataset Ratios & Best Practices
- **Industry Standards**:
  - Positive:Negative ratio = 1:5 to 1:10
  - Hard negatives = 20-30% of negatives
  - Background noise mixing at 30-50% probability
  - RIR augmentation at 20-30% probability
- Train/val/test split ratios
- Balancing strategies
- Augmentation guidelines

#### 4. Configuration Parameters Explained
- **Basic Parameters**:
  - sample_rate: Why 16kHz? (balance quality and speed)
  - audio_duration: Typical wakeword length (1-2 seconds)
  - batch_size: GPU memory considerations
  - learning_rate: Starting values and scheduling

- **Advanced Parameters**:
  - Augmentation parameters and their effects
  - Optimizer choices (Adam recommended)
  - Loss functions (focal loss for imbalance)
  - Scheduler strategies
  - Mixed precision benefits

#### 5. Model Architectures
- Architecture comparison table
- When to use each architecture
- Accuracy vs speed vs model size tradeoffs
- Transfer learning benefits

#### 6. Training Process
- Training stages explained
- Interpreting metrics:
  - Accuracy: Overall correctness
  - FPR: False alarms (critical to minimize)
  - FNR: Missed detections (user frustration)
- Early stopping and overfitting
- Troubleshooting common issues

#### 7. Evaluation & Deployment
- Threshold selection guidelines
- Real-world testing strategies
- Edge deployment considerations
- ONNX model optimization

#### 8. Troubleshooting Guide
- **Low Accuracy**: Insufficient data, poor quality, imbalanced classes
- **High FPR**: Need more hard negatives, lower threshold
- **High FNR**: Need more positive samples, raise threshold
- **Overfitting**: Too complex model, insufficient augmentation
- **GPU OOM**: Reduce batch size, shorter audio duration
- **Slow Training**: Check num_workers, data loading bottleneck

#### 9. Glossary
- Technical terms defined
- Acronyms explained (FPR, FNR, SNR, RIR, etc.)

**UI Implementation**:
- Markdown renderer (Gradio Markdown component)
- Searchable content
- Collapsible sections
- Code examples with syntax highlighting
- Links to external resources (papers, tutorials)

---

## Phase 8: Integration & Testing

### 8.1 Gradio Application
**File**: `src/ui/app.py`

**Features**:
- Tab-based interface with 6 panels
- Shared state management between panels
- Configuration persistence (save/load across sessions)
- Error handling and user-friendly messages
- Responsive design
- Dark/light theme support
- Avaliable port selection ability between 7860-7870 ports

**Launch Configuration**:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    inbrowser=True
)
```

### 8.2 Testing Strategy
- Unit tests for core functions (pytest)
- GPU/CUDA availability tests
- Integration tests for full pipeline
- Performance benchmarks (training speed, inference speed)

### 8.3 Error Handling
- GPU out-of-memory handling
- Audio file corruption handling
- Invalid configuration detection
- No CPU fallback for tensor related training processes, Hard Rule!
- User-friendly and explanatory error messages

---

## Phase 9: Documentation & Deployment

### 9.1 User Documentation
**Files**: `README.md`, `docs/`

**README.md Contents**:
- Installation instructions (CUDA setup)
- Quick start guide
- System requirements
- Troubleshooting
- License and acknowledgments

### 9.2 Code Documentation
- Docstrings for all functions/classes (Google style)
- Type hints throughout codebase
- Inline comments for complex logic
- Architecture diagrams

### 9.3 Deployment Considerations
- Docker containerization option
- Requirements.txt with version pinning
- Installation validation script
- GPU driver compatibility notes

---

## Implementation Order (Recommended)

### Sprint 1: Foundation (Week 1)
1. Project structure setup
2. Requirements and environment
3. Basic Gradio app skeleton with 6 empty panels
4. CUDA detection and validation
5. Logging infrastructure

### Sprint 2: Dataset Management (Week 1-2)
1. Dataset scanner and validator
2. Train/test/val splitting logic
3. .npy extractor
4. Panel 1 UI implementation
5. Health briefing report

### Sprint 3: Configuration (Week 2)
1. Default configuration definitions
2. Configuration validator
3. Panel 2 UI with basic/advanced tabs
4. Save/load configuration

### Sprint 4: Training Pipeline (Week 2-3)
1. PyTorch Dataset class
2. Data augmentation pipeline
3. Model architectures implementation
4. Training loop with GPU support
5. Metrics tracking

### Sprint 5: Training UI (Week 3)
1. Panel 3 UI implementation
2. Live plotting integration
3. Training control (start/pause/stop)
4. Progress indicators
5. Checkpoint management

### Sprint 6: Evaluation (Week 4)
1. File-based evaluation
2. Microphone inference
3. Panel 4 UI implementation
4. Metrics visualization

### Sprint 7: Export & Docs (Week 4)
1. ONNX converter
2. Panel 5 UI
3. Knowledge base content writing
4. Panel 6 UI with markdown rendering

### Sprint 8: Integration & Polish (Week 5)
1. End-to-end testing
2. Error handling refinement
3. Performance optimization
4. Documentation completion
5. README and setup guides

---

## Key Technical Decisions

### GPU/CUDA Strategy
- **Device Management**: Automatic GPU detection, no fallback to CPU for tensor processes!
- **Memory Optimization**:
  - Gradient checkpointing for large models
  - Mixed precision training (FP16)
  - Memory-mapped file loading for large datasets
  - Batch size auto-tuning based on GPU memory
- **Multi-GPU**: DataParallel for single-node (optional future feature)

### Audio Processing Pipeline
- **Format Standardization**: Convert all to 16kHz 16-bit mono WAV on load
- **Feature Extraction**: On-the-fly during training (GPU-accelerated)
- **Augmentation**: GPU-based with torchaudio transforms
- **Caching**: Optional feature caching to disk for repeated experiments

### Data Loading Optimization
- **Parallel Loading**: num_workers = max(cpu_count -8 , 8)
- **Prefetching**: 2x batch_size prefetch buffer
- **Pin Memory**: Enabled for faster GPU transfer
- **Persistent Workers**: Reuse worker processes across epochs

### Model Selection Rationale
- **ResNet-18**: Proven architecture, good accuracy, reasonable speed
- **MobileNetV3**: Best for edge deployment, real-time inference
- **LSTM/GRU**: Good for sequential patterns, slower training
- **TCN**: Modern choice, parallel training, competitive accuracy

### Evaluation Metrics Priority
1. **False Positive Rate (FPR)**: Most critical for user experience
2. **False Negative Rate (FNR)**: Important but less critical than FPR
3. **Accuracy**: Overall performance indicator
4. **Latency**: Inference speed for real-time applications

---

## Industry Standards & Best Practices

### Dataset Requirements
- **Minimum Positive Samples**: 1000 utterances (more is better)
- **Negative:Positive Ratio**: 8:1 to 10:1
- **Hard Negatives**: 20-30% of total negatives
- **Speaker Diversity**: 50+ unique speakers for generalization
- **Recording Quality**: 16kHz+, 16-bit, low background noise
- **Duration**: Consistent length (1.5-2 seconds typical)

### Augmentation Guidelines
- **Time Stretch**: ±20% (0.8-1.2x)
- **Pitch Shift**: ±2 semitones
- **Background Noise**: SNR 5-20 dB, 30-50% probability
- **RIR Convolution**: 20-30% probability
- **Gaussian Noise**: SNR 20-40 dB, 20% probability

### Training Hyperparameters (Defaults)
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Batch Size**: 32-128 (GPU memory dependent)
- **Epochs**: 50-100 with early stopping
- **Weight Decay**: 1e-4
- **Label Smoothing**: 0.1 (helps generalization)
- **Gradient Clipping**: 1.0 (prevents exploding gradients)

### Performance Targets
- **Accuracy**: >95% on test set
- **FPR**: <5% (ideally <2%)
- **FNR**: <5%
- **Inference Latency**: <100ms (real-time)
- **Model Size**: <50MB (edge deployment)

---

## Risk Mitigation

### Performance Risks
- **300k Files Processing**:
  - Risk: Memory overflow, slow loading
  - Mitigation: Memory-mapped files, streaming, batch processing

- **GPU OOM During Training**:
  - Risk: Batch size too large
  - Mitigation: Automatic batch size finder, gradient accumulation

### Data Quality Risks
- **Imbalanced Dataset**:
  - Risk: Model bias toward majority class
  - Mitigation: Weighted sampling, class weights, focal loss

- **Poor Audio Quality**:
  - Risk: Low model performance
  - Mitigation: Quality filtering, automated checks, user warnings

### User Experience Risks
- **Complex Configuration**:
  - Risk: User confusion, poor results
  - Mitigation: Presets, validation, extensive documentation

- **Long Training Times**:
  - Risk: User impatience, resource waste
  - Mitigation: Progress indicators, ETA, pause/resume functionality

---

## Success Metrics

### Functional Completeness
- ✅ All 6 panels implemented and functional
- ✅ GPU/CUDA acceleration mandatory
- ✅ 300k files processed without crashes
- ✅ All dataset types supported
- ✅ ONNX export successful

### Performance Benchmarks
- ✅ Training speed: >1000 samples/sec on modern GPU
- ✅ Inference latency: <50ms per sample
- ✅ Model accuracy: >95% on balanced test set
- ✅ FPR: <5%

### Usability
- ✅ Setup time: <30 minutes (including CUDA)
- ✅ Documentation clarity: No external help needed
- ✅ Error messages: Clear and actionable
- ✅ UI responsiveness: <1 second for interactions

---

## Future Enhancements (Post-MVP)

### Advanced Features
- Speaker embedding integration
- Multi-wakeword support
- Online learning / continual training
- Automatic hyperparameter tuning (Optuna)
- Federated learning support

### Deployment Tools
- Model compression (pruning, quantization)
- TensorFlow Lite export
- iOS/Android model formats
- C++ inference engine
- REST API for inference

### UI Improvements
- Multi-language support
- Custom model architecture builder
- Experiment tracking (MLflow integration)
- Distributed training support

---

## Open Questions for Review

1. **Model Architecture**: Any preference among ResNet-18, MobileNet, LSTM, TCN?
2. **Feature Type**: Mel-spectrogram or MFCC as input? (Recommend Mel-spec)
3. **Augmentation Intensity**: Conservative or aggressive augmentation?
4. **Deployment Target**: Edge device constraints? (affects model choice)
5. **Real-time Requirements**: Streaming inference needed or batch only?
6. **Multi-wakeword**: Future support for multiple wakewords simultaneously?
7. **Pre-trained Models**: Include pre-trained model zoo for fine-tuning?

---

## Estimated Timeline

- **Sprint 1-2 (Foundation + Dataset)**: 1.5 weeks
- **Sprint 3-4 (Config + Training Pipeline)**: 1.5 weeks
- **Sprint 5-6 (Training UI + Evaluation)**: 1.5 weeks
- **Sprint 7-8 (Export + Docs + Polish)**: 1.5 weeks

**Total**: ~6 weeks for complete, production-ready implementation

---

## Dependencies & Prerequisites

### System Requirements
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- CUDA Toolkit 11.8 or 12.x
- cuDNN 8.x
- Python 3.8-3.11
- 16GB+ RAM (for large datasets)
- 100GB+ disk space (for 300k audio files)

### Python Packages (requirements.txt)
```
# Core
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
sounddevice>=0.4.0
scipy>=1.10.0

# Data & ML
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0

# Export
onnx>=1.14.0
onnxruntime-gpu>=1.15.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.13.0
```

---

## Appendix: Technical References

### Wakeword Detection Papers
- "Honk: A PyTorch Reimplementation of Convolutional Neural Networks for Keyword Spotting" (2017)
- "Hello Edge: Keyword Spotting on Microcontrollers" (2019)
- "Temporal Convolution for Real-time Keyword Spotting" (2019)

### Audio Augmentation
- SpecAugment: A Simple Data Augmentation Method for ASR
- Room Impulse Response (RIR) databases: MIT, AIR, RIRS_NOISES

### Model Architectures
- ResNet: "Deep Residual Learning for Image Recognition"
- MobileNet: "MobileNets: Efficient CNNs for Mobile Vision"
- TCN: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks"

---

## Notes
- All file paths are placeholders and will be created during implementation
- Configuration values are industry-standard defaults, fully customizable
- GPU memory estimates assume RTX 3080+ (10GB+ VRAM)
- Timeline assumes single developer, full-time work
- Testing with 300k files may require dataset sampling during development

---

**Status**: Ready for review and feedback
**Next Step**: Await user edits/approval, then proceed with implementation assignments