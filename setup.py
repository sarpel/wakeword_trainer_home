"""
Wakeword Training Platform Setup
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wakeword-training-platform",
    version="1.0.0",
    author="Wakeword Platform Team",
    description="GPU-accelerated wakeword training platform with Gradio UI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.1",
        "torchaudio>=2.0.2",
        "gradio>=3.50.2",
        "librosa>=0.10.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "onnx>=1.14.1",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.1",
    ],
)