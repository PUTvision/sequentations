import io
import os
import re
from setuptools import setup, find_packages


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "sequentations", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


try:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        REQUIRED = f.read().split('\n')
except:
    REQUIRED = []

setup(
    name="sequentations",
    version=get_version(),
    description="Albumentations based library for sequence of images",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="PUT Vision",
    license="MIT",
    url="https://github.com/PUTvision/sequentations",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=REQUIRED,
    extras_require={"tests": ["pytest"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
