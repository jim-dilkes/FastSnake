from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("backend/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="snakebench",
    version="0.1.0",
    author="Greg Kamradt",
    author_email="greg@arcprize.org",
    description="A competitive snake game simulation environment for benchmarking LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gkamradt/SnakeBench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "snakebench": [
            "backend/model_lists/*",
            "backend/completed_games/*",
        ]
    },
    entry_points={
        "console_scripts": [
            "snakebench=snakebench.backend.main:main",
            "snakebench-elo=snakebench.backend.elo_tracker:main",
        ],
    }
)
