from setuptools import setup, find_packages

setup(
    name="ma_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    author="00Dvs00",
    description="M&A Analysis and Reporting System",
    python_requires=">=3.9",
)