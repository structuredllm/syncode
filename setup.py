import setuptools
python_requires=">=3.6,<3.13",

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the content of the requirements.txt file without mxeval
requirements = [
    "fire",
    "interegular",
    "regex==2023.8.8",
    "torch",
    "tqdm",
    "transformers==4.44.0",
    "datasets",
    "jsonschema"
]

setuptools.setup(
    name="syncode",
    version="0.4.10",
    author="Shubham Ugare",
    author_email="shubhamugare@gmail.com",
    description="This package provides the tool for grammar augmented LLM generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/structuredllm/syncode",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "syncode.evaluation.mxeval": ["*.json", "*.py"],
    },
)