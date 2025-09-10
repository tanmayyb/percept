# from mp_eval.assets.procedural.wall_generator import WallGenerator
from mp_eval.assets.procedural.walls_generator import WallsGenerator

import yaml
import argparse

TOWER_CONFIG = [
"""
dense_wall1:
    thickness: 4
    spacing: 0.040
    plane: "xz"
    wall_width: 0.25
    wall_height: 1.0
    # wall_center_i: 0.75
    wall_center_i: 0.50
    wall_center_j: 0.50
    wall_offset_k: 0.0
    include_hole: false
"""
]


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


def generate_towers(args):
    assets_configs = TOWER_CONFIG

    generated_assets = []

    for asset_config in assets_configs:
        scene_generation_params = yaml.safe_load(asset_config)
        walls_generator = WallsGenerator(scene_generation_params)
        generated_assets.append(walls_generator.generate_procedurally())
    
    merged_assets = []
    for asset_data in generated_assets:
        merged_assets.extend(asset_data)


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