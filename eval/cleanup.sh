#!/bin/bash

# Get the directory from command line argument
dir=$1

if [ -z "$dir" ]; then
    echo "Please provide a directory path as argument"
    exit 1
fi

# Check if directory exists
if [ ! -d "$dir" ]; then
    echo "Directory '$dir' does not exist"
    exit 1
fi

# Find and remove all .yaml files in the directory
find "$dir" -name "*.yaml" -type f -delete

echo "Cleaned up YAML files in $dir"
