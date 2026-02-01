"""Setup script for Fashion MNIST Classifier."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="fashion-mnist-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready Fashion MNIST image classifier with CNN, API, and web interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fashion-mnist-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0,<3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "gradio>=4.0.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fashion-mnist-train=train:main",
            "fashion-mnist-predict=predict:main",
            "fashion-mnist-api=api.main:main",
        ],
    },
)
