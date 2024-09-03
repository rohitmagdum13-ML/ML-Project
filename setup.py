import setuptools
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

NAME = "ML Project"
AUTHOR_USER_NAME = "Rohit"
AUTHOR_EMAIL = "rohitmagdum1306@gmail.com"


setuptools.setup(
    name=NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),      # Automatically find and include all packages in the project
    include_package_data=True,     # Include additional files listed in MANIFEST.in
)