import cupoch as cph
import yaml
import argparse
import numpy as np
import math
from copy import deepcopy
import ast

CONCAVE_CONFIG = [
"""
radius: 0.25
paraboloid_apex_z: 0.15
paraboloid_a: 2.50
spacing: 0.040
"""
]


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
        title='concave Bowl Scene',
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
        f.write('# generated concave\n')
        data = {"obstacles": assets}
        yaml.dump(data, f, default_flow_style=None)
    print(f"Scene saved to {filepath}")




def bowl_pointcloud(params: dict = None) -> np.ndarray:
    params = params or {}
    dtype = np.float64

    radius        = params.get('radius', 1.0)
    paraboloid_a  = params.get('paraboloid_a', 0.50)   # curvature
    apex_z        = params.get('paraboloid_apex_z', radius)  # c in z = c - a r^2
    grid_size     = params.get('spacing', 0.040)

    # Uniform xyz grid
    x = np.arange(-radius, radius + 1e-12, grid_size, dtype=dtype)
    y = np.arange(-radius, radius + 1e-12, grid_size, dtype=dtype)
    z = np.arange(0.0,      radius + 1e-12, grid_size, dtype=dtype)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r2_xy = X*X + Y*Y

    # Upper hemisphere
    inside_hemisphere = (r2_xy + Z*Z) <= (radius * radius)

    # Downward-opening paraboloid: z = apex_z - a * (x^2 + y^2)
    z_parab = apex_z - paraboloid_a * r2_xy

    # Keep the shell between hemisphere (outer) and paraboloid (inner)
    mask = inside_hemisphere & (Z >= z_parab)

    pts = np.stack((X[mask], Y[mask], Z[mask]), axis=1).astype(dtype, copy=False)
    return pts


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



def main():
    parser = argparse.ArgumentParser(description='Generate a concave bowl scene')
    parser.add_argument('--plot', action='store_true', help='Plot the generated scene using plotly')
    parser.add_argument('--coloraxis', type=str, choices=['x', 'y', 'z'], 
                       help='Specify axis (x, y, or z) to use for color scaling in the plot')
    parser.add_argument('--dump', action='store_true', help='Dump the generated scene in cmdline')
    parser.add_argument('--save', action='store_true', help='Save the generated scene to a file')    
    
    parser.add_argument('--shape_params', type=str, help='Variables for the scene configuration')
    parser.add_argument('--tf_params', type=str, help='Variables for the scene configuration')

    args = parser.parse_args()

    # load default params
    scene_configs = yaml.safe_load(CONCAVE_CONFIG[0])
    pts_config = {
        'scale': 1.00,
        'rot': 'X+1/2,Z+1/4',
        'trans': '0.75,0.0,0.25'
    }

    if args.shape_params:
        vars_dict = ast.literal_eval(args.shape_params)
        mapping = {'r': 'radius', 'a': 'paraboloid_a', 'b': 'paraboloid_apex_z', 's': 'spacing'}
        for k, v in vars_dict.items():
            if k in mapping:
                scene_configs[mapping[k]] = v

    if args.tf_params:
        vars_dict = ast.literal_eval(args.tf_params)
        mapping = {'s': 'scale', 'r': 'rot', 't': 'trans'}
        for k, v in vars_dict.items():
            if k in mapping:
                pts_config[mapping[k]] = v



    pts = bowl_pointcloud(params=scene_configs)
    pts = deepcopy(tf.transform_objects(pts, pts_config))
    obstacles = convert_pointcloud_to_obstacles(pts)

    # Plot if requested
    if args.plot:
        plot_scene(obstacles, args)
    
    # Save if requested
    if args.save:
        save_scene(obstacles)


if __name__ == "__main__":
    main()