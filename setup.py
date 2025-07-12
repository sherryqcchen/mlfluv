from setuptools import setup, find_packages

setup(
    name="mlfluv",
    version="0.1.0",
    description="A Python package for Planet data ordering and processing.",
    author="Qiuyang Chen",
    author_email="qiuyangschen@gmail.com",
    packages=find_packages(),  # Automatically find all packages in the project
    install_requires=[
        "requests",
        "geopandas",
        "pandas",
        "rasterio",
        "matplotlib",
        "planet"
    ],
    python_requires=">=3.8",
)