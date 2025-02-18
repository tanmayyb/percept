#!/usr/bin/env python3

import random
import yaml
def generate_maze_yaml(num_walls=5,           # Number of parallel walls
                      wall_width=21,          # Number of obstacles across each wall
                      wall_thickness=1,       # Number of rows of obstacles per wall
                      spacing=0.50,           # Distance between obstacles
                      obstacle_radius=0.25,   # Radius of each obstacle
                      gate_width=2,           # Width of gates in obstacle units
                      wall_spacing=2.0,       # Distance between parallel walls
                      origin_x=0.0,           # X coordinate of maze origin
                      origin_y=0.0,           # Y coordinate of maze origin
                      randomize_gates=True,   # If True, randomize gate positions
                      seed=None):            # Random seed for reproducibility
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Create the YAML content as a string
    yaml_content = '# 2D maze with parallel walls and gates\nobstacles:\n'
    
    for wall_idx in range(num_walls):
        # Calculate base wall position
        wall_y_base = wall_idx * wall_spacing + origin_y
        
        # Determine gate position for this wall
        if randomize_gates:
            # Calculate margin (10% of wall width on each side)
            margin = int(wall_width * 0.2)
            # Restrict gate position to middle 80% of wall
            gate_center = random.randint(
                margin + gate_width//2, 
                wall_width - margin - gate_width//2
            )
        else:
            gate_center = wall_width // 2  # Center of the wall
            
        gate_min = gate_center - gate_width // 2
        gate_max = gate_center + gate_width // 2
        
        # Generate obstacles for each row in the wall
        for thickness_idx in range(wall_thickness):
            wall_y = wall_y_base + thickness_idx * spacing
            
            # Generate obstacles across the wall
            for x in range(wall_width):
                # Skip positions where the gate should be
                if gate_min <= x <= gate_max:
                    continue
                    
                x_pos = round(x * spacing + origin_x, 2)
                y_pos = round(wall_y, 2)
                z_pos = 0.0
                
                # Format position entry
                yaml_content += f'- position: [{x_pos}, {y_pos}, {z_pos}]\n'
                if obstacle_radius != 0.25:  # Only include radius if different from default
                    yaml_content += f'  radius: {obstacle_radius}\n'
    
    # Write to file
    path = 'src/percept_core/assets/benchmark_scenes/maze_2d.yaml'
    with open(path, 'w') as f:
        f.write(yaml_content)

if __name__ == '__main__':
    # Example: Create a maze with 5 walls, each having a random gate position
    # generate_maze_yaml(
    #     num_walls=3,
    #     wall_width=51,
    #     wall_thickness=2,
    #     spacing=0.100,
    #     obstacle_radius=0.025,  # Using default radius
    #     gate_width=3,
    #     wall_spacing=1.0,
    #     origin_x=-2.50,
    #     origin_y=1.0,
    #     randomize_gates=True,
    #     seed=42  # Set seed for reproducibility
    # )


    generate_maze_yaml(
        num_walls=3,
        wall_width=101,
        wall_thickness=2,
        spacing=0.050,
        obstacle_radius=0.025,  # Using default radius
        gate_width=6,
        wall_spacing=1.0,
        origin_x=-2.50,
        origin_y=1.0,
        randomize_gates=True,
        seed=42  # Set seed for reproducibility
    )
