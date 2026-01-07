"""
Setup script for Cyberbullying Detection Project

Allows for editable installation:
    pip install -e .

This makes all project modules importable without path manipulation.
"""

from setuptools import setup, find_packages
import os

# Read requirements.txt
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        # Filter out comments, empty lines, and conditional dependencies
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove inline comments
                line = line.split('#')[0].strip()
                # Skip conditional dependencies (they should be installed separately if needed)
                if ';' not in line:
                    requirements.append(line)
                # For conditional dependencies, keep the full line with condition
                # setuptools will handle them appropriately
                else:
                    requirements.append(line)
        return requirements

setup(
    name='cyberbullying-detection',
    version='0.1.0',
    description='Kannada-English Code-Mixed Cyberbullying Detection System',
    author='Cyberbullying Detection Project Team',
    author_email='vinod.hn@example.com',
    url='https://github.com/vinod-hn/Cyberbullying-detection',
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    python_requires='>=3.8',
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='cyberbullying detection nlp machine-learning kannada code-mixed',
    project_urls={
        'Source': 'https://github.com/vinod-hn/Cyberbullying-detection',
        'Bug Reports': 'https://github.com/vinod-hn/Cyberbullying-detection/issues',
    },
)
