"""
ONNX Model Exporter
Convert PyTorch models to ONNX with quantization and optimization
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass
import time
import logging
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None
    logging.warning("ONNX/ONNXRuntime not installed. Export functionality limited.")

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for ONNX export"""
    output_path: Path
    opset_version: int = 14
    dynamic_batch: bool = True
    quantize_fp16: bool = False
    quantize_int8: bool = False
    optimize: bool = True
    verbose: bool = False


class ONNXExporter:
    """
    Export PyTorch models to ONNX format with optimization
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input_shape: Tuple[int, ...],
        device: str = 'cuda'
    ):
        """
        Initialize ONNX exporter

        Args:
            model: PyTorch model to export
            sample_input_shape: Shape of sample input (e.g., (1, 1, 64, 50))
            device: Device for model
        """
        if onnx is None or ort is None:
            raise ImportError(
                "ONNX and ONNXRuntime required. "
                "Install with: pip install onnx onnxruntime onnxruntime-gpu"
            )

        self.model = model
        self.sample_input_shape = sample_input_shape
        self.device = device

        # Move model to device and eval mode
        self.model.to(device)
        self.model.eval()

        logger.info(f"ONNXExporter initialized with input shape: {sample_input_shape}")

    def export(
        self,
        config: ExportConfig
    ) -> Dict[str, Any]:
        """
        Export model to ONNX

        Args:
            config: Export configuration

        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting model to ONNX: {config.output_path}")

        # Create output directory
        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(*self.sample_input_shape).to(self.device)

        # Dynamic axes for dynamic batch size
        dynamic_axes = None
        if config.dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Export to ONNX
        logger.info("Converting PyTorch model to ONNX...")

        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(config.output_path),
                export_params=True,
                opset_version=config.opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=config.verbose
            )

            logger.info(f"✅ ONNX model exported to: {config.output_path}")

            # Get file size
            file_size_mb = config.output_path.stat().st_size / (1024 * 1024)

            results = {
                'success': True,
                'path': str(config.output_path),
                'opset_version': config.opset_version,
                'dynamic_batch': config.dynamic_batch,
                'file_size_mb': file_size_mb
            }

            # Apply quantization if requested
            if config.quantize_fp16:
                logger.info("Applying FP16 quantization...")
                fp16_path = self._quantize_fp16(config.output_path)
                fp16_size_mb = fp16_path.stat().st_size / (1024 * 1024)
                results['fp16_path'] = str(fp16_path)
                results['fp16_size_mb'] = fp16_size_mb
                results['fp16_reduction'] = (1 - fp16_size_mb / file_size_mb) * 100

            if config.quantize_int8:
                logger.info("Applying INT8 quantization...")
                int8_path = self._quantize_int8(config.output_path, dummy_input)
                int8_size_mb = int8_path.stat().st_size / (1024 * 1024)
                results['int8_path'] = str(int8_path)
                results['int8_size_mb'] = int8_size_mb
                results['int8_reduction'] = (1 - int8_size_mb / file_size_mb) * 100

            return results

        except Exception as e:
            logger.error(f"Export failed: {e}")
            logger.exception(e)
            return {
                'success': False,
                'error': str(e)
            }

    def _quantize_fp16(self, onnx_path: Path) -> Path:
        """
        Apply FP16 quantization to ONNX model

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            output_path = onnx_path.parent / f"{onnx_path.stem}_fp16.onnx"

            # FP16 quantization (weight-only)
            quantize_dynamic(
                str(onnx_path),
                str(output_path),
                weight_type=QuantType.QFloat16,
                optimize_model=True
            )

            logger.info(f"✅ FP16 model saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"FP16 quantization failed: {e}")
            raise

    def _quantize_int8(self, onnx_path: Path, calibration_data: torch.Tensor) -> Path:
        """
        Apply INT8 quantization to ONNX model

        Args:
            onnx_path: Path to ONNX model
            calibration_data: Sample data for calibration

        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            output_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"

            # INT8 quantization (dynamic)
            quantize_dynamic(
                str(onnx_path),
                str(output_path),
                weight_type=QuantType.QInt8,
                optimize_model=True
            )

            logger.info(f"✅ INT8 model saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            raise


def export_model_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    quantize_fp16: bool = False,
    quantize_int8: bool = False,
    device: str = 'cuda'
) -> Dict:
    """
    Export PyTorch model checkpoint to ONNX

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output ONNX path
        opset_version: ONNX opset version
        dynamic_batch: Allow dynamic batch size
        quantize_fp16: Apply FP16 quantization
        quantize_int8: Apply INT8 quantization
        device: Device for model

    Returns:
        Dictionary with export results
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain configuration")

    config = checkpoint['config']

    # Create model
    from src.models.architectures import create_model

    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=False,
        dropout=config.model.dropout
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {config.model.architecture}")

    # Determine input shape based on config
    # Assuming mel-spectrogram input
    sample_rate = config.data.sample_rate
    duration = config.data.audio_duration
    n_mels = config.data.n_mels
    hop_length = config.data.hop_length

    # Calculate time steps
    n_samples = int(sample_rate * duration)
    n_frames = n_samples // hop_length + 1

    # Input shape: (batch, channels, n_mels, n_frames)
    sample_input_shape = (1, 1, n_mels, n_frames)

    logger.info(f"Input shape: {sample_input_shape}")

    # Create exporter
    exporter = ONNXExporter(model, sample_input_shape, device)

    # Create export config
    export_config = ExportConfig(
        output_path=output_path,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        quantize_fp16=quantize_fp16,
        quantize_int8=quantize_int8,
        optimize=True,
        verbose=False
    )

    # Export
    results = exporter.export(export_config)

    # Add model info
    results['architecture'] = config.model.architecture
    results['sample_rate'] = sample_rate
    results['duration'] = duration
    results['input_shape'] = sample_input_shape

    return results


def validate_onnx_model(
    onnx_path: Path,
    pytorch_model: Optional[nn.Module] = None,
    sample_input: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Validate ONNX model

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Optional PyTorch model for comparison
        sample_input: Sample input for testing
        device: Device for computation

    Returns:
        Dictionary with validation results
    """
    if onnx is None or ort is None:
        raise ImportError("ONNX and ONNXRuntime required")

    logger.info(f"Validating ONNX model: {onnx_path}")

    results = {
        'valid': False,
        'error': None
    }

    try:
        # Load and check ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        results['valid'] = True
        results['graph'] = len(onnx_model.graph.node)
        results['inputs'] = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                            for i in onnx_model.graph.input]
        results['outputs'] = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                             for o in onnx_model.graph.output]

        # Get model size
        file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        results['file_size_mb'] = file_size_mb

        # Test inference if sample input provided
        if sample_input is not None:
            logger.info("Testing ONNX inference...")

            # ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess = ort.InferenceSession(str(onnx_path), providers=providers)

            # Get input name
            input_name = sess.get_inputs()[0].name

            # Run inference
            onnx_input = {input_name: sample_input.cpu().numpy()}
            onnx_output = sess.run(None, onnx_input)[0]

            results['inference_success'] = True
            results['output_shape'] = onnx_output.shape

            # Compare with PyTorch if provided
            if pytorch_model is not None:
                pytorch_model.eval()
                pytorch_model.to(device)

                with torch.no_grad():
                    pytorch_output = pytorch_model(sample_input.to(device)).cpu().numpy()

                # Calculate difference
                max_diff = np.abs(pytorch_output - onnx_output).max()
                mean_diff = np.abs(pytorch_output - onnx_output).mean()

                results['max_difference'] = float(max_diff)
                results['mean_difference'] = float(mean_diff)
                results['numerically_equivalent'] = max_diff < 1e-3

                logger.info(f"Max difference: {max_diff:.6f}")
                logger.info(f"Mean difference: {mean_diff:.6f}")

        logger.info("✅ ONNX model validation complete")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        results['valid'] = False
        results['error'] = str(e)

    return results


def benchmark_onnx_model(
    onnx_path: Path,
    pytorch_model: nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 100,
    device: str = 'cuda'
) -> Dict:
    """
    Benchmark ONNX model vs PyTorch

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: PyTorch model for comparison
        sample_input: Sample input tensor
        num_runs: Number of inference runs
        device: Device for computation

    Returns:
        Dictionary with benchmark results
    """
    if ort is None:
        raise ImportError("ONNXRuntime required")

    logger.info(f"Benchmarking ONNX model (n={num_runs})...")

    results = {}

    # Benchmark PyTorch
    pytorch_model.eval()
    pytorch_model.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = pytorch_model(sample_input.to(device))

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = pytorch_model(sample_input.to(device))

    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs * 1000  # ms

    results['pytorch_time_ms'] = pytorch_time

    # Benchmark ONNX
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    input_name = sess.get_inputs()[0].name
    onnx_input = {input_name: sample_input.cpu().numpy()}

    # Warmup
    for _ in range(10):
        _ = sess.run(None, onnx_input)

    # Benchmark
    start_time = time.time()

    for _ in range(num_runs):
        _ = sess.run(None, onnx_input)

    onnx_time = (time.time() - start_time) / num_runs * 1000  # ms

    results['onnx_time_ms'] = onnx_time
    results['speedup'] = pytorch_time / onnx_time
    results['num_runs'] = num_runs

    logger.info(f"PyTorch: {pytorch_time:.2f} ms")
    logger.info(f"ONNX: {onnx_time:.2f} ms")
    logger.info(f"Speedup: {results['speedup']:.2f}x")

    return results


if __name__ == "__main__":
    # Test ONNX export
    print("ONNX Export Module Test")
    print("=" * 60)

    if onnx is None or ort is None:
        print("⚠️  ONNX/ONNXRuntime not installed")
        print("Install with: pip install onnx onnxruntime onnxruntime-gpu")
    else:
        print(f"✅ ONNX version: {onnx.__version__}")
        print(f"✅ ONNXRuntime version: {ort.__version__}")

        # Check available providers
        print("\nAvailable Execution Providers:")
        for provider in ort.get_available_providers():
            print(f"  - {provider}")

    print("\nONNX export module ready")
