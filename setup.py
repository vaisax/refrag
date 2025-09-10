from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="refrag-simple",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simplified implementation of REFRAG for efficient RAG decoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/refrag-simple",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "refrag-demo=refrag:main",
        ],
    },
    keywords="retrieval-augmented generation, RAG, language models, NLP, efficiency, compression",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/refrag-simple/issues",
        "Source": "https://github.com/yourusername/refrag-simple",
        "Documentation": "https://github.com/yourusername/refrag-simple#readme",
        "Original Paper": "https://arxiv.org/abs/2509.01092",
    },
)