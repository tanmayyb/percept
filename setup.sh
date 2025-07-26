#!/bin/bash

set -e  # Exit on any error

ENV_NAME="percept_env"
ENV_FILE="environment.yml"
CURRENT_DIR="$(pwd)"

# Set up PERCEPT_ROOT in .bashrc
if ! grep -q "export PERCEPT_ROOT=" ~/.bashrc || ! grep -q "export PERCEPT_ROOT=$CURRENT_DIR" ~/.bashrc; then
    echo "🔧 Updating PERCEPT_ROOT in ~/.bashrc..."
    sed -i '/export PERCEPT_ROOT=/d' ~/.bashrc  # Remove any existing definition
    echo "export PERCEPT_ROOT=$CURRENT_DIR" >> ~/.bashrc
    echo "✅ PERCEPT_ROOT set to: $CURRENT_DIR"
else
    echo "ℹ️ PERCEPT_ROOT already correctly set in ~/.bashrc"
fi

# Initialize Conda shell integration
eval "$(conda shell.bash hook)" || { echo "❌ Failed to initialize conda"; exit 1; }
conda init bash

# Create the conda environment only if it doesn't already exist
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "✅ Conda environment '$ENV_NAME' already exists, skipping creation."
else
    echo "🚀 Creating conda environment from $ENV_FILE..."
    conda env create -f "$ENV_FILE" || { echo "❌ Failed to create environment"; exit 1; }
fi

# Activate the environment
echo "⚙️  Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME" || { echo "❌ Failed to activate conda environment"; exit 1; }

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

echo "✅ Setup complete. You may need to restart your terminal for PERCEPT_ROOT to take effect."
