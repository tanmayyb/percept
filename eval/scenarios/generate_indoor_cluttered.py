"""
Generate an Indoor Cluttered Scenario using the ShapeNetPart dataset.

ShapeNetPartNet dataset available at:
https://www.kaggle.com/datasets/majdouline20/shapenetpart-dataset

Run the script from this directory:
cd percept/eval/scenarios
python generate_indoor_cluttered.py
"""


import cupoch as cph
import os
from pathlib import Path
import json
import numpy as np
from copy import deepcopy

root_path = Path(os.getcwd())
shape_net_path = root_path / "ShapeNetPart" / "PartAnnotation"
shape_net_index_path = shape_net_path / "metadata.json"

with open(shape_net_index_path, "r") as f:
  shape_net_index = json.load(f)


def load_pointcloud(dict_key) -> cph.geometry.PointCloud:
  global shape_net_index, shape_net_path
  dict_value = shape_net_index[dict_key]
  directory = dict_value["directory"]
  lables = dict_value["lables"]
  colors = dict_value["colors"]

  # load the pointcloud
  pointcloud_dirpath = shape_net_path / directory / "points"

  points = []

  for file in pointcloud_dirpath.iterdir():
    if file.is_file() and file.suffix == ".pts":
      with open(file, "r") as f:
        for line in f:
          coords = list(map(float, line.strip().split()))
          points.append(coords)
    break

  np_points = np.array(points)
  pc = cph.geometry.PointCloud()
  pc.points = cph.utility.Vector3fVector(np_points)
  return pc

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
  def transform_objects(object_config):
    """
    Parse the transformation configuration and apply transformations to objects.
    
    Args:
        object_config (dict): Dictionary containing transformation configurations for each object

    Returns:
        dict: Dictionary containing the transformed pointcloud objects
    """
    transformed_objects = {}
    objects = {}
   
    for obj_name, config in object_config.items():

      try:
        objects[obj_name] = tf.rotX(load_pointcloud(obj_name), np.pi/2)
      except:
        print(f"Warning: Object {obj_name} not found in objects dictionary")
        continue
        
      pc = objects[obj_name]
      
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
          
      transformed_objects[obj_name] = pc
        
    return transformed_objects

from mp_eval.assets.procedural.walls_generator import WallsGenerator
import yaml

room_dim = np.array([3.0, 5.0, 1.5])
room_center = np.array([0.0, 0.0, 0.0])

def generate_room_walls():
  # Define room dimensions
  global room_dim, room_center

  room_walls_config = {
      "wall_floor": {
          "spacing": 0.04,
          "plane": "xy",
          "wall_width": room_dim[0],
          "wall_height": room_dim[1],
          "wall_center_i": 0.0,
          "wall_center_j": 0.0,
          "wall_offset_k": room_center[2] - room_dim[2] / 2.0,
          "include_hole": False
      },
      "wall_ceiling": {
          "spacing": 0.04,
          "plane": "xy",
          "wall_width": room_dim[0],
          "wall_height": room_dim[1],
          "wall_center_i": 0.0,
          "wall_center_j": 0.0,
          "wall_offset_k": room_center[2] + room_dim[2] / 2.0,
          "include_hole": False
      },
      "wall_front": {
          "spacing": 0.04,
          "plane": "xz",
          "wall_width": room_dim[0],
          "wall_height": room_dim[2],
          "wall_center_i": 0.0,
          "wall_center_j": 0.0,
          "wall_offset_k": room_center[1] + room_dim[1] / 2.0,
          "include_hole": False
      },
      "wall_left": {
          "spacing": 0.04,
          "plane": "yz",
          "wall_width": room_dim[2],
          "wall_height": room_dim[1],
          "wall_center_i": 0.0,
          "wall_center_j": 0.0,
          "wall_offset_k": room_center[0] - room_dim[0] / 2.0,
          "include_hole": False
      }
  }

  walls_pc = WallsGenerator(room_walls_config).generate_procedurally(generate_raw=True)
  walls_pc = np.array(walls_pc)
  pc = cph.geometry.PointCloud()
  pc.points = cph.utility.Vector3fVector(walls_pc)

  return pc

room_pc = generate_room_walls()

object_config = {
  'Chair': {
    'scale': 1.75,
    'rot': 'Z-1/2',
    'trans': '0.0,1.25,-0.40'
  },
  'Table': {
    'scale': 2.0,
    'rot': 'Z-1/2',
    'trans': '0.0,0.0,-0.25'
  },
  'Laptop':{
    'scale': 1.0,
    'rot': 'Z1/2',
    'trans': '0.0,0.0,0.0'
  },
  'Mug': {
    'scale': 0.25,
    'trans': '-0.70,0.75,-0.10'
  },
  'Bag': {
    'trans': '1.0,0.0,-0.45'
  },
  'Lamp': {
    'scale': 1.50,
    'rot': 'Y1/2,Z1/2',
    'trans': '1.0,2.00,-0.10'
  },
  'Guitar': {
    'scale': 1.5,
    'rot': 'Z-1/2,X-1/6',
    'trans': '-1.10,2.15,-0.25'
  },
  'Airplane': {
    'scale': 2.0,
    'trans': '0.0,1.0,0.35'
  },
  'Rocket': {
    'scale': 2.0,
    'trans': '0.0,-1.0,0.45'
  },
  'Motorbike': {
    'scale': 2.25,
    'trans': '0.40,-1.33,-0.40'
  },
  'Car': {
    'scale': 2.5,
    'rot': 'Z-1/2',
    'trans': '-1.07,-1.45,-0.25'
  },
}
objects = tf.transform_objects(object_config)
# cph.visualization.draw_geometries([room_pc] + list(objects.values()))


import cupy as cp
import numba.cuda as cuda



voxel_size = 0.020

class UniformScene:
   @staticmethod
   def generate_uniform_scene():
      global room_pc, objects, room_center, room_dim, voxel_size
      total_pc = deepcopy(room_pc)
      for obj in objects.values():
        total_pc += obj


      voxel_grid = cph.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        total_pc,
        voxel_size=voxel_size,
        min_bound=[room_center[0] - room_dim[0] / 2.0, room_center[1] - room_dim[1] / 2.0, room_center[2] - room_dim[2] / 2.0],
        max_bound=[room_center[0] + room_dim[0] / 2.0, room_center[1] + room_dim[1] / 2.0, room_center[2] + room_dim[2] / 2.0],
      )

      # start = time.time()
      voxels = voxel_grid.voxels.cpu()
      primitives_pos = np.array(list(voxels.keys()))

      if primitives_pos.size == 0:  # Handle empty voxel grid
        print("No voxels found in voxel grid")
        return None
      
      # Transfer data to GPU
      primitives_pos_gpu = cp.asarray(primitives_pos)
      offset = cp.asarray(voxel_grid.get_min_bound())
      voxel_size = cp.asarray(voxel_size)
      
      # Compute minimums for each column on GPU
      mins = cp.min(primitives_pos_gpu, axis=0)
      
      # Perform operations on GPU (Normalize and scale)
      # Subtract mins: Shifts all voxel indices so the minimum is at zero (origin).
      # Scale: Multiplies by the voxel size to convert from grid indices to real-world coordinates.
      # Offset: Adds the minimum bound and half a voxel size to center each primitive in its voxel.
      primitives_pos_gpu = primitives_pos_gpu - mins[None, :]
      primitives_pos_gpu = primitives_pos_gpu * voxel_size
      primitives_pos_gpu = primitives_pos_gpu + (offset + voxel_size/2)

      # save copy of primitives_pos_gpu
      primitives_pos_gpu = cuda.as_cuda_array(primitives_pos_gpu)
      
      # Transfer result back to CPU
      primitives_pos = cp.asnumpy(primitives_pos_gpu)

      pc = cph.geometry.PointCloud()
      pc.points = cph.utility.Vector3fVector(primitives_pos)

      return voxel_grid, primitives_pos, pc


voxel_grid, primitives_pos, pc = UniformScene.generate_uniform_scene()
# cph.visualization.draw_geometries([pc])


def convert_to_wall(primitives_pos):
  obstacles = []

  for primitive in primitives_pos:
    obstacles.append({
      # 'name': f"Obstacle{obstacle_id}",
      'position': [round(float(p), 2) for p in primitive],
      # 'velocity': [0.0, 0.0, 0.0],
      'radius': 0.020,
      # 'is_dynamic': False,
      # 'angular_speed': 0.0,
    })
  
  obstacles_data = {
    'obstacles': obstacles
  }
  
  return obstacles_data
obstacles_data = convert_to_wall(primitives_pos)

yaml_path = root_path / "../../src/percept_core/assets/benchmark_scenes/indoor_cluttered.yaml"
with open(yaml_path, "w") as f:
  yaml.dump(obstacles_data, f, sort_keys=False)