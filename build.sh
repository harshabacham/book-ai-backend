#!/bin/bash
# Force Python version check
python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 required'"

# Clean installation
python -m pip install --upgrade pip wheel setuptools cython==0.29.36

# Install requirements
python -m pip install --no-build-isolation --no-cache-dir -r requirements.txt
