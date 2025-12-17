from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl-vlm-lab",
    version="0.1.0",
    author="RL-VLM-Lab Contributors",
    description="Preference tuning framework for Vision-Language Models using DPO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RL-VLM-Lab",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.40.0",
        "datasets>=4.4.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "huggingface-hub>=0.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "tensorboard": ["tensorboard>=2.13.0"],
    },
    include_package_data=True,
    zip_safe=False,
)