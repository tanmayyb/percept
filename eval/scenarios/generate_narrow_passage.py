from mp_eval.assets.procedural.wall_generator import WallGenerator
import yaml
import argparse

WALL1_SCENE_PARAMS_YAML = """
spacing: 0.10
plane: "xz"
wall_width: 6.0
wall_height: 6.0
wall_center_i: 0.0
wall_center_j: 0.0
wall_offset_k: 0.5
include_hole: true
hole_center_x: -0.5
hole_center_y: 0.0
hole_width: 1.0
hole_height: 6.0
"""

WALL2_SCENE_PARAMS_YAML = """
spacing: 0.10
plane: "xz"
wall_width: 6.0
wall_height: 6.0
wall_center_i: 0.0
wall_center_j: 0.0
wall_offset_k: 1.0
include_hole: true
hole_center_x: 0.5
hole_center_y: 0.0
hole_width: 1.0
hole_height: 6.0
"""



def plot_scene(generated_assets, args):
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
    pio.templates.default = "plotly_white"
    
    # Combine all points from generated assets
    all_points = []
    for asset_data in generated_assets:
        all_points.extend(asset_data)
    
    # Extract x, y, z coordinates
    x = [point['position'][0] for point in all_points]
    y = [point['position'][1] for point in all_points]
    z = [point['position'][2] for point in all_points]
    
    # Create 3D scatter plot
    marker_dict = dict(
        size=2,
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
        title='Narrow Passage Scene',
        scene=dict(
            aspectmode='data'
        )
    )
    
    fig.show()


def save_scene(generated_assets):
    from pathlib import Path
    from ament_index_python.packages import get_package_share_directory
    filepath = Path(get_package_share_directory('percept')) / "assets/benchmark_scenes/auto_generated_scene.yaml"
    with open(filepath, 'w') as f:
        yaml.dump(generated_assets, f)
    print(f"Scene saved to {filepath}")


def generate_narrow_passage(args):
    assets_configs = [
        WALL1_SCENE_PARAMS_YAML,
        WALL2_SCENE_PARAMS_YAML,
    ]

    generated_assets = []
    for asset_config in assets_configs:
        scene_generation_params = yaml.safe_load(asset_config)
        wall_generator = WallGenerator(scene_generation_params)
        generated_assets.append(wall_generator.generate_procedurally())
    
    if args.dump:
        data = []
        for asset_data in generated_assets:
            data.extend(asset_data)
        print(data)
    
    if args.plot:
        plot_scene(generated_assets, args)
    
    if args.save:
        save_scene(generated_assets)
    
    return

def main():
    parser = argparse.ArgumentParser(description='Generate a narrow passage scene')
    parser.add_argument('--plot', action='store_true', help='Plot the generated scene using plotly')
    parser.add_argument('--coloraxis', type=str, choices=['x', 'y', 'z'], 
                       help='Specify axis (x, y, or z) to use for color scaling in the plot')
    parser.add_argument('--dump', action='store_true', help='Dump the generated scene in cmdline')
    parser.add_argument('--save', action='store_true', help='Save the generated scene to a file')
    args = parser.parse_args()
    
    generate_narrow_passage(args)


if __name__ == "__main__":
    main()