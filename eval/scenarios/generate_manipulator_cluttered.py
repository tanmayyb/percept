# from mp_eval.assets.procedural.wall_generator import WallGenerator
from mp_eval.assets.procedural.walls_generator import WallsGenerator

import yaml
import argparse

import cupoch as cph
import numpy as np
from copy import deepcopy


shape_configs = [
"""
dense_wall1:
    thickness: 7
    spacing: 0.040
    plane: "xz"
    wall_width: 0.50
    wall_height: 1.0 
    wall_center_i: 0.0
    wall_center_j: 0.50
    wall_offset_k: 0.0
    include_hole: false
    hole_center_x: 0.0
    hole_center_y: 0.0
    hole_width: 0.25
    hole_height: 0.30
""",
# """
# wall1:
#     spacing: 0.040
#     plane: "xy"
#     wall_width: 2.0
#     wall_height: 2.0 
#     wall_center_i: 0.0
#     wall_center_j: 0.0
#     wall_offset_k: 0.0
#     include_hole: false
# """
]


tf_configs = {
    'dense_wall1':{
        'scale':1.0,
        # 'rot':'',
        'trans':'0.50,0.0,0.0',

    }
}



class tf:
  @staticmethod
  def rotX(pointcloud, angle):
    matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    return deepcopy(pointcloud).rotate(matrix)

  @staticmethod
  def rotY(pointcloud, angle):
    matrix = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return deepcopy(pointcloud).rotate(matrix)

  @staticmethod
  def rotZ(pointcloud, angle):
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return deepcopy(pointcloud).rotate(matrix)

  @staticmethod
  def trans(pointcloud, translation):
    matrix = np.array([[1, 0, 0, translation[0]], [0, 1, 0, translation[1]], [0, 0, 1, translation[2]], [0, 0, 0, 1]])
    return deepcopy(pointcloud).transform(matrix)
  
  @staticmethod
  def scale(pointcloud, scale):
    return deepcopy(pointcloud).scale(scale, True)

  @staticmethod
  def transform_objects(np_points, config):
    pc = cph.geometry.PointCloud()
    pc.points = cph.utility.Vector3fVector(np_points)

    # Apply scale if specified
    if 'scale' in config:
        pc = tf.scale(pc, config['scale'])
        
    # Apply rotations if specified
    if 'rot' in config:
        rot_str = config['rot']
        # Split rotation string into individual rotations
        rotations = rot_str.split(',')
            
        for rot in rotations:
            axis = rot[0].upper()  # X, Y, or Z
            # Parse the angle value
            angle_str = rot[1:]
            if '/' in angle_str:
                num, denom = angle_str.split('/')
                angle = np.pi * float(num) / float(denom)
            else:
                angle = np.pi * float(angle_str)
                
            # Apply the rotation based on axis
            if axis == 'X':
                pc = tf.rotX(pc, angle)
            elif axis == 'Y':
                pc = tf.rotY(pc, angle)
            elif axis == 'Z':
                pc = tf.rotZ(pc, angle)

    # Apply translation if specified
    if 'trans' in config:
        trans_str = config['trans']
        translation = [float(x) for x in trans_str.split(',')]
        pc = tf.trans(pc, translation)
        
    return np.array(pc.points.cpu())


def plot_scene(assets, args):
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
    pio.templates.default = "plotly_white"
    
    # Combine all points from generated assets
    x = [point['position'][0] for point in assets]
    y = [point['position'][1] for point in assets]
    z = [point['position'][2] for point in assets]
    
    x += [0]
    y += [0]
    z += [0]
    
    # Create 3D scatter plot
    marker_dict = dict(
        size=8,
        color='red',
    )
    
    if args.coloraxis:
        marker_dict['color'] = y if args.coloraxis == 'y' else x if args.coloraxis == 'x' else z
        marker_dict['colorscale'] = 'Viridis'
        marker_dict['colorbar'] = dict(
            title=f'{str(args.coloraxis).upper()} Position',
            thickness=20,
            len=0.75
        )

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=marker_dict
    )])
    
    fig.update_layout(
        title='Towers Scene',
        scene=dict(
            aspectmode='data'
        )
    )
    
    fig.show()


def save_scene(assets):
    from pathlib import Path
    from ament_index_python.packages import get_package_share_directory
    filepath = Path(get_package_share_directory('percept')) / "assets/benchmark_scenes/auto_generated_scene.yaml"
    with open(filepath, 'w') as f:
        f.write('# generated towers\n')
        data = {"obstacles": assets}
        yaml.dump(data, f, default_flow_style=None)
    print(f"Scene saved to {filepath}")


def convert_pointcloud_to_obstacles(pointcloud):
    obstacles = []
    
    for point in pointcloud:
        obstacles.append({
            'position': [round(float(point[0]), 2), 
                        round(float(point[1]), 2), 
                        round(float(point[2]), 2)],
            # 'radius': obstacle_radius
        })
    
    return obstacles

def generate_towers(args):
    assets_configs = shape_configs

    generated_assets = {}

    for asset_config in assets_configs:
        scene_generation_params = yaml.safe_load(asset_config)
        object_key = list(scene_generation_params.keys())[0]
        walls_generator = WallsGenerator(scene_generation_params)
        generated_assets[object_key] = walls_generator.generate_procedurally(generate_raw=True)

    for object_key, tf_config in tf_configs.items():
        generated_assets[object_key] = tf.transform_objects(generated_assets[object_key], tf_config)

    merged_assets = []
    for asset_list in generated_assets.values():
        merged_assets.extend(asset_list)
    merged_assets = np.array(merged_assets)

    merged_assets = convert_pointcloud_to_obstacles(merged_assets)

    if args.dump:
        print(merged_assets)
    
    if args.plot:
        plot_scene(merged_assets, args)
    
    if args.save:
        save_scene(merged_assets)
    
    return

def main():
    parser = argparse.ArgumentParser(description='Generate a towers scene')
    parser.add_argument('--plot', action='store_true', help='Plot the generated scene using plotly')
    parser.add_argument('--coloraxis', type=str, choices=['x', 'y', 'z'], 
                       help='Specify axis (x, y, or z) to use for color scaling in the plot')
    parser.add_argument('--dump', action='store_true', help='Dump the generated scene in cmdline')
    parser.add_argument('--save', action='store_true', help='Save the generated scene to a file')
    args = parser.parse_args()
    
    generate_towers(args)


if __name__ == "__main__":
    main()