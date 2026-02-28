from setuptools import setup, find_packages

setup(
    name="salsa_unlearn",
    version="0.1.0",
    author="Your Name",
    description="A comprehensive library for Machine Unlearning methods, including SALSA.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/unlearn_lib",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "scikit-learn>=0.24.2",
        "xgboost",
        "numpy",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)