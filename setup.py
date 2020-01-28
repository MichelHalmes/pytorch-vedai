from setuptools import setup, find_packages


version = open("src/version").read()
long_description = open("README.md").read()

install_require = [
    "torch >=1.3, <1.4",
    "torchvision >=0.4, <0.5",
    "matplotlib >=3.1, <4",
    "tensorboard >=2.0, <3",
    "click >=7.0, <8",
    "pillow >=6.2, <7",
    "opencv-python >=4.1, <5",
]

scripts_require = [
    "psutil >=5.6, <6"
]

setup_args = {
    "name": "pytorch_vedai",
    "version": version,
    "author": "Michel Halmes",
    "author_email": "none",
    "python_requires": ">= 3",
    "description": "Applying object detection to satellite images using pyTorch",
    "long_description": long_description,
    "url": "https://github.com/MichelHalmes/pytorch-vedai",
    "packages": find_packages(include=["src"]),
    "package_dir": {"src": "src"},
    "package_data": {"": ["*.md", "version"]},
    "install_requires": install_require,
    "extras_require": {"scripts": scripts_require},
    "entry_points": {
        "console_scripts": [
            "train_distr=src.entrypoints.train_distributed:main",
            "run_eval=src.entrypoints.run_evaluation:main",
            "convert_images=scripts.convert_images:main",
            "demo_augmentation=scripts.demo_augmentation:main",
            "demo_memory_leak=scripts.demo_memory_leak:main",
        ],
    }
}

setup(**setup_args)
