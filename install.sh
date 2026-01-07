#!/bin/bash
set -e

echo "Installing omnynet-export..."

# Check for pip
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

# Install from GitHub
pip install git+https://github.com/wcares/omnynet-export.git

echo "Done! Run 'omnynet-export --help' to get started."
