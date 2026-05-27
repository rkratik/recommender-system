from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="recommender-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GenAI-Powered Movie Recommender System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rkratik/recommender-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
    },
)
