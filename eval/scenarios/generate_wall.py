#!/usr/bin/env python3
from typing import List, Dict
import argparse
import datetime

"""
Wall Generator for Percept

Follows the following (PMAF) format:

obstacles:
- name: "Obstacle1"
  position: [5.0, 5.0, 0.0]
  velocity: [0.0, 0.0, 0.0]
  radius: 0.5
  is_dynamic: true
  angular_speed: 0.5
"""

class WallGenerator:
    def __init__(
            self, 
            scene_config_params:dict,
            params_check:list = [
                "spacing",
            ]
        ):
        self.params_check = params_check
        self.scene_config_params = scene_config_params
        self.invalid_config = False

        self._validate_config()
        if self.invalid_config:
            raise ValueError("Invalid configuration for wall generation")


    def _validate_config(self):
        params_check = self.params_check
        for param in params_check:
            if param not in self.scene_config_params:
                self.invalid_config = True
                return False
        return True
    
    # def _load_config(self):
    #     self.spacing = self.scene_config_params.get('spacing', 1.0)
    #     self.plane = self.scene_config_params.get('plane', 'xz')
    #     self.wall_width = self.scene_config_params.get('wall_width')
    #     self.wall_height = self.scene_config_params.get('wall_height')
    #     self.wall_center_i = self.scene_config_params.get('wall_center_i', 0.0)
    #     self.wall_center_j = self.scene_config_params.get('wall_center_j', 0.0)
    #     self.wall_offset_k = self.scene_config_params.get('wall_offset_k', 0.0)

    #     self.include_hole = self.scene_config_params.get('include_hole', False)
    #     self.hole_center_x = self.scene_config_params.get('hole_center_x', 0.0) # wall frame
    #     self.hole_center_y = self.scene_config_params.get('hole_center_y', 0.0) # wall frame
    #     self.hole_width = self.scene_config_params.get('hole_width', 0.0)
    #     self.hole_height = self.scene_config_params.get('hole_height', 0.0)

    def _load_config(self):
        self.spacing = self.scene_config_params.get('spacing', 1.0)
        self.plane = self.scene_config_params.get('plane', 'xz')
        self.wall_width = self.scene_config_params.get('wall_width')
        self.wall_height = self.scene_config_params.get('wall_height')
        self.wall_center_i = self.scene_config_params.get('wall_center_i', 0.0)
        self.wall_center_j = self.scene_config_params.get('wall_center_j', 0.0)
        self.wall_offset_k = self.scene_config_params.get('wall_offset_k', 0.0)
        self.include_hole = self.scene_config_params.get('include_hole', False)
        self.hole_center_x = self.scene_config_params.get('hole_center_x', 0.0) # wall frame
        self.hole_center_y = self.scene_config_params.get('hole_center_y', 0.0) # wall frame
        self.hole_width = self.scene_config_params.get('hole_width', 0.0)
        self.hole_height = self.scene_config_params.get('hole_height', 0.0)
        self.radius = self.scene_config_params.get('radius', 0.5)

    def _generate_wall(self):
        # Compute the wall's lower left corner in world coordinates
        wall_start_i = self.wall_center_i - self.wall_width / 2.0
        wall_start_j = self.wall_center_j - self.wall_height / 2.0

        # Determine the number of grid cells along each dimension.
        num_cells_i = int(round(self.wall_width / self.spacing))
        num_cells_j = int(round(self.wall_height / self.spacing))
        
        obstacles = []

        # Precompute the hole bounds in world coordinates (if the hole is enabled)
        if self.include_hole:
            # The hole's center is given relative to the wall's center
            hole_world_center_i = self.wall_center_i + self.hole_center_x
            hole_world_center_j = self.wall_center_j + self.hole_center_y
            hole_half_width = self.hole_width / 2.0
            hole_half_height = self.hole_height / 2.0
            hole_min_i = hole_world_center_i - hole_half_width
            hole_max_i = hole_world_center_i + hole_half_width
            hole_min_j = hole_world_center_j - hole_half_height
            hole_max_j = hole_world_center_j + hole_half_height

        obstacle_id = 1
        # Loop over grid cells to create the wall
        for cell_i in range(num_cells_i):
            for cell_j in range(num_cells_j):
                # Compute the center position of the current grid cell in world coordinates
                cell_pos_i = wall_start_i + (cell_i + 0.5) * self.spacing
                cell_pos_j = wall_start_j + (cell_j + 0.5) * self.spacing
                
                # If the hole is enabled and the cell falls within the hole bounds, skip it.
                if self.include_hole:
                    if hole_min_i <= cell_pos_i <= hole_max_i and hole_min_j <= cell_pos_j <= hole_max_j:
                        continue

                # Choose the correct coordinate ordering based on the selected plane
                if self.plane == 'xz':  # Wall parallel to ground (X-Z plane)
                    pos = [cell_pos_i, self.wall_offset_k, cell_pos_j]
                elif self.plane == 'xy':  # Vertical wall along the X axis (X-Y plane)
                    pos = [cell_pos_i, cell_pos_j, self.wall_offset_k]
                elif self.plane == 'yz':  # Vertical wall along the Y axis (Y-Z plane)
                    pos = [self.wall_offset_k, cell_pos_j, cell_pos_i]
                else:
                    pos = [cell_pos_i, self.wall_offset_k, cell_pos_j]

                # - name: "Obstacle1"
                #     position: [5.0, 5.0, 0.0]
                #     velocity: [0.0, 0.0, 0.0]
                #     radius: 0.5
                #     is_dynamic: true
                #     angular_speed: 0.5

                obstacles.append({
                    'name': f"Obstacle{obstacle_id}",
                    'position': [round(p, 2) for p in pos],
                    'velocity': [0.0, 0.0, 0.0],
                    'radius': self.radius,
                    'is_dynamic': False,
                    'angular_speed': 0.0,
                })
                obstacle_id += 1

        wall_data = {
            'obstacles': obstacles
        }
        return wall_data


    def generate_procedurally(self) -> List[Dict]:
        self._load_config()
        return self._generate_wall()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a wall configuration.",
        epilog=(
            "Example usage:\n"
            "  python generate_wall.py --wall_width 5 --wall_height 3 --spacing 1.0 --plane xz \\\n"
            "    --wall_center_i 0 --wall_center_j 0 --wall_offset_k 0 \\\n"
            "    --include_hole --hole_center_x 0 --hole_center_y 0 --hole_width 1 --hole_height 1 \\\n"
            "    --radius 0.5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--spacing', type=float, default=1.0, help='Spacing between wall elements')
    parser.add_argument('--plane', type=str, default='xz', choices=['xz', 'xy', 'yz'], help='Plane of the wall')
    parser.add_argument('--wall_width', type=float, required=True, help='Width of the wall')
    parser.add_argument('--wall_height', type=float, required=True, help='Height of the wall')
    parser.add_argument('--wall_center_i', type=float, default=0.0, help='Wall center coordinate i')
    parser.add_argument('--wall_center_j', type=float, default=0.0, help='Wall center coordinate j')
    parser.add_argument('--wall_offset_k', type=float, default=0.0, help='Wall offset in k direction')
    parser.add_argument('--include_hole', action='store_true', help='Whether to include a hole in the wall')
    parser.add_argument('--hole_center_x', type=float, default=0.0, help='Hole center x (wall frame)')
    parser.add_argument('--hole_center_y', type=float, default=0.0, help='Hole center y (wall frame)')
    parser.add_argument('--hole_width', type=float, default=0.0, help='Width of the hole')
    parser.add_argument('--hole_height', type=float, default=0.0, help='Height of the hole')
    parser.add_argument('--radius', type=float, default=0.5, help='Radius of the obstacle')

    args = parser.parse_args()

    config = {
        'spacing': args.spacing,
        'plane': args.plane,
        'wall_width': args.wall_width,
        'wall_height': args.wall_height,
        'wall_center_i': args.wall_center_i,
        'wall_center_j': args.wall_center_j,
        'wall_offset_k': args.wall_offset_k,
        'include_hole': args.include_hole,
        'hole_center_x': args.hole_center_x,
        'hole_center_y': args.hole_center_y,
        'hole_width': args.hole_width,
        'hole_height': args.hole_height,
        'radius': args.radius,
    }

    wall_gen = WallGenerator(config)
    wall_data = wall_gen.generate_procedurally()

    import yaml
    with open(f"{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}_wall.yaml", "w") as f:
        yaml.dump(wall_data, f, sort_keys=False)
