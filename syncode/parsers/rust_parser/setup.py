from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="rust_parser",  # Just the specific module name
    version="0.1.0",
    rust_extensions=[
        RustExtension(
            "rust_parser",  # Simple module name
            binding=Binding.PyO3,
            path="./Cargo.toml",  # Path relative to setup.py in this directory
        )
    ],
    packages=find_packages(),  # This will find all packages with __init__.py
    package_dir={"": "."},  # This tells setuptools the root directory
    zip_safe=False,
)