# 🎙️ Wakeword Training Platform

GPU-accelerated platform for training custom wakeword detection models with a comprehensive Gradio UI.

## ✨ Features

- **Complete Training Pipeline**: Dataset management → Training → Evaluation → Export
- **6 Intuitive Panels**: Step-by-step workflow with Gradio interface
- **GPU-Accelerated**: CUDA-powered training with PyTorch (CPU fallback not supported)
- **Industry Standards**: Best practices built-in with configurable parameters
- **Real-time Monitoring**: Live metrics, plots, and GPU utilization
- **Model Export**: ONNX export with quantization options
- **Comprehensive Documentation**: Built-in knowledge base and troubleshooting

## 📋 System Requirements

### Mandatory
- **NVIDIA GPU** with CUDA support (Compute Capability 6.0+)
- **CUDA Toolkit** 11.8 or 12.x
- **cuDNN** 8.x
- **Python** 3.8-3.11

### Recommended
- **GPU Memory**: 8GB+ (10GB+ for large datasets)
- **RAM**: 16GB+
- **Storage**: 100GB+ for datasets
- **CPU**: Multi-core processor for data loading

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd wakeword-training-platform

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### 2. Prepare Your Datasets

Organize your audio files into the following structure:

```
data/raw/
├── positive/           # Your wakeword utterances
├── negative/           # Non-wakeword speech
├── hard_negative/      # Similar-sounding phrases
├── background/         # Environmental noise
├── rirs/              # Room impulse responses
└── npy/               # Pre-computed features (optional)
```

**Supported Formats**: WAV, MP3, FLAC, OGG

**Requirements**:
- Minimum 1,000 positive samples (5,000+ recommended)
- Negative:Positive ratio of 8:1 to 10:1
- 16kHz sample rate, 16-bit, mono
- 1.5-2 seconds duration

### 3. Launch the Platform

```bash
python src/ui/app.py
```

Or:

```bash
python -m src.ui.app
```

The interface will open automatically at `http://localhost:7860` (or next available port 7860-7870).

## 📊 Workflow

### Panel 1: Dataset Management
1. Select dataset root directory
2. Scan and validate datasets
3. Review health report
4. Split into train/validation/test sets
5. Extract .npy files if available

### Panel 2: Configuration
1. Choose preset or customize parameters
2. Configure basic settings (sample rate, batch size, epochs)
3. Fine-tune advanced options (augmentation, optimizer)
4. Save/load configurations

### Panel 3: Training
1. Start training with configured parameters
2. Monitor real-time metrics (loss, accuracy, FPR, FNR)
3. View live plots and GPU utilization
4. Pause/stop training as needed
5. Best model automatically saved

### Panel 4: Evaluation
1. Load trained model
2. Test with audio files
3. Use microphone for live testing
4. Run test set evaluation
5. Analyze metrics and confusion matrix

### Panel 5: ONNX Export
1. Select model checkpoint
2. Configure export options
3. Enable quantization (FP16/INT8)
4. Export and validate
5. Download ONNX model

### Panel 6: Documentation
Complete knowledge base covering:
- Dataset preparation guidelines
- Configuration parameter explanations
- Training best practices
- Troubleshooting guide
- Industry standards
- Glossary

## 🎯 Key Metrics

### Training Targets
- **Accuracy**: >95% on test set
- **False Positive Rate (FPR)**: <5% (ideally <2%)
- **False Negative Rate (FNR)**: <5%
- **Training Speed**: >1000 samples/sec on modern GPU
- **Inference Latency**: <50ms per sample

### Industry Standards
- **Positive Samples**: 1,000+ (5,000+ recommended)
- **Negative:Positive Ratio**: 8:1 to 10:1
- **Hard Negatives**: 20-30% of negatives
- **Speaker Diversity**: 50+ unique voices
- **Audio Quality**: 16kHz, 16-bit, mono

## 🔧 Configuration Presets

| Preset | Use Case | Characteristics |
|--------|----------|-----------------|
| Default | General purpose | Balanced settings |
| Small Dataset | <10k samples | Aggressive augmentation |
| Large Dataset | >100k samples | Optimized for scale |
| Fast Training | Quick iteration | Minimal augmentation |
| High Accuracy | Best performance | Extended training |
| Edge Deployment | Mobile/IoT | Optimized for size |

## 📁 Project Structure

```
wakeword-training-platform/
├── src/
│   ├── config/         # Configuration and utilities
│   ├── data/           # Dataset processing
│   ├── models/         # Model architectures
│   ├── training/       # Training logic
│   ├── evaluation/     # Evaluation tools
│   ├── export/         # ONNX export
│   └── ui/             # Gradio interface
├── data/
│   ├── raw/            # Original datasets
│   ├── processed/      # Processed splits
│   └── splits/         # Split manifests
├── models/
│   ├── checkpoints/    # Saved models
│   └── exported/       # ONNX models
├── logs/               # Training logs
├── docs/               # Documentation
├── requirements.txt    # Dependencies
├── verify_installation.py  # Setup verification
└── README.md
```

## 🛠️ Troubleshooting

### CUDA Not Available
```bash
# Check GPU and CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA (for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors
- Reduce batch size in Panel 2
- Use smaller audio duration
- Choose lighter model (MobileNetV3)
- Enable gradient accumulation

### Poor Training Performance
- Check dataset balance (health report in Panel 1)
- Add more diverse data
- Increase augmentation
- Review configuration parameters
- Consult troubleshooting guide in Panel 6

### Installation Issues
```bash
# Run verification script
python verify_installation.py

# Create fresh virtual environment
python -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

## 📚 Documentation

Comprehensive documentation is available within the platform:
- Navigate to **Panel 6: Documentation**
- Topics include:
  - Dataset preparation guide
  - Configuration parameters explained
  - Training best practices
  - Evaluation strategies
  - Troubleshooting guide
  - Complete glossary

## 🎓 Recommended Reading

- **Wakeword Detection Papers**:
  - "Honk: A PyTorch Reimplementation of CNNs for Keyword Spotting" (2017)
  - "Hello Edge: Keyword Spotting on Microcontrollers" (2019)

- **Audio Augmentation**:
  - SpecAugment: A Simple Data Augmentation Method for ASR

- **Model Architectures**:
  - ResNet: Deep Residual Learning for Image Recognition
  - MobileNets: Efficient CNNs for Mobile Vision

## ⚙️ Advanced Usage

### Custom Model Architectures
Models are defined in `src/models/architectures.py`. You can add custom architectures by extending the base classes.

### Custom Augmentations
Augmentation pipeline is in `src/data/augmentation.py`. Add new transforms to the pipeline.

### Multi-GPU Training
Multi-GPU support can be enabled by setting `DataParallel` in training configuration (future feature).

## 📝 Development Status

### Current Version: 1.0 (Sprint 1 Complete)
✅ Project structure and environment setup
✅ CUDA detection and validation
✅ Logging infrastructure
✅ 6-panel Gradio interface skeleton
✅ Installation verification script
✅ Comprehensive documentation

### Upcoming (Sprint 2+)
⏳ Dataset processing implementation
⏳ Configuration management
⏳ Training pipeline
⏳ Real-time monitoring
⏳ Evaluation tools
⏳ ONNX export

## 🤝 Contributing

This is a production-focused, reliability-first implementation. When contributing:

1. Follow existing code structure
2. Maintain GPU-only policy (no CPU fallback)
3. Add comprehensive error handling
4. Update documentation
5. Test thoroughly with real datasets

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- PyTorch team for excellent deep learning framework
- Gradio team for intuitive UI framework
- Research community for wakeword detection advancements

## 📞 Support

For issues and questions:
1. Check Panel 6 Documentation (Troubleshooting section)
2. Run `python verify_installation.py`
3. Review logs in `logs/` directory
4. Consult implementation plan

## 🎯 Quick Reference

### Launch Platform
```bash
python src/ui/app.py
```

### Verify Setup
```bash
python verify_installation.py
```

### Check GPU Status
```python
from src.config.cuda_utils import enforce_cuda
cuda_validator = enforce_cuda()
print(cuda_validator.get_device_info())
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

**Built with ❤️ for reliable, production-grade wakeword training**