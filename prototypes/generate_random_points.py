#!/usr/bin/env python3

import random
import yaml

def generate_random_points(
    num_points=20,
    x_range=(2.0, 8.0),
    y_range=(2.0, 8.0),
    z_range=(-1.5, 1.8),
    output_path='obstacles_random.yaml',
    seed=None
):
    """
    Generate random obstacle points within specified ranges.
    
    Args:
        num_points: Number of obstacle points to generate
        x_range: Tuple of (min, max) for x coordinates
        y_range: Tuple of (min, max) for y coordinates
        z_range: Tuple of (min, max) for z coordinates
        output_path: Path to save the YAML file
    """

    if seed is not None:
        random.seed(seed)
    
    obstacles = []
    
    for _ in range(num_points):
        x = round(random.uniform(*x_range), 2)
        y = round(random.uniform(*y_range), 2)
        z = round(random.uniform(*z_range), 2)
        
        obstacles.append({
            'position': [x, y, z]
        })
    
    # Create the YAML structure
    data = {'obstacles': obstacles}
    
    # Write to YAML file
    with open(output_path, 'w') as f:
        f.write('# random points example\n')
        yaml.dump(data, f, default_flow_style=None)

if __name__ == '__main__':
    # Example usage with parameters similar to obstacles3.yaml
    generate_random_points(
        num_points=20,
        x_range=(-0.25, 0.25),
        y_range=(0.5, 2.0),
        z_range=(0.0, 2.0),
        output_path='src/percept_core/config/obstacles3.yaml',
        seed=42
    )