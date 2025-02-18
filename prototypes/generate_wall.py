#!/usr/bin/env python3

def generate_wall_yaml(wall_width=21, wall_height=21, spacing=0.50, # radius = 1/2 of spacing
                      hole_center_x=5.0, hole_center_z=5.0, hole_width=2.0, hole_height=2.0,
                      wall_origin_x=0.0, wall_origin_z=0.0):
    # Convert hole center from meters to grid coordinates
    hole_grid_x = hole_center_x / spacing
    hole_grid_z = hole_center_z / spacing
    
    # Calculate hole bounds
    hole_x_min = hole_grid_x - hole_width / 2
    hole_x_max = hole_grid_x + hole_width / 2
    hole_z_min = hole_grid_z - hole_height / 2
    hole_z_max = hole_grid_z + hole_height / 2

    path = 'src/percept_core/config/hole_in_the_wall_hard.yaml'
    with open(path, 'w') as f:
        f.write('# hole in the wall example\n')
        f.write('obstacles:\n')
        
        # Generate grid of obstacles
        for x in range(wall_width):
            for z in range(wall_height):
                x_pos = x * spacing + wall_origin_x
                z_pos = z * spacing + wall_origin_z
                
                # Skip if position is within hole bounds (using float comparison)
                if (hole_x_min <= x <= hole_x_max and 
                    hole_z_min <= z <= hole_z_max):
                    continue
                    
                y_pos = 1.0  # Fixed height
                
                # Write obstacle entry with rounded positions
                f.write(f'  - position: [{x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f}]\n')
                f.write('    radius: 0.25\n')

if __name__ == '__main__':
    # Example: Create a wall with a hole in the middle, translated 2m in X and 1m in Z
    generate_wall_yaml(
        # wall_width=21,
        # wall_height=21,
        # spacing=0.10,
        # hole_center_x=1.00,
        # hole_center_z=1.00,
        # hole_width=5,
        # hole_height=5,
        # wall_origin_x=-1.0,
        # wall_origin_z=0.0

        # even harder wall
        wall_width=21,
        wall_height=21,
        spacing=0.10,
        hole_center_x=1.33,
        hole_center_z=1.00,
        hole_width=2,
        hole_height=4,
        wall_origin_x=-1.0,
        wall_origin_z=0.0

        # # medium wall values
        # wall_width=21,
        # wall_height=21,
        # spacing=0.10,
        # hole_center_x=1.33,
        # hole_center_z=1.00,
        # hole_width=2,
        # hole_height=6,
        # wall_origin_x=-1.0,
        # wall_origin_z=0.0
    )