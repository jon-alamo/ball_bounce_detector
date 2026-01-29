from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="bounce_detector",
    version="0.1.0",
    packages=find_packages(include=["bounce_detector", "bounce_detector.*"]),
    install_requires=required,
    include_package_data=True,
    package_data={
        "bounce_detector": ["assets/*.pkl"]
    },
    description="A library to detect padel ball bounces from 2D coordinates.",
    long_description=open("README.md").read() if "README.md" in open("README.md").read() else "",
    long_description_content_type="text/markdown",
    author="Joni",
    python_requires=">=3.8",
)
