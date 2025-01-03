#!/usr/bin/env python3

import rospy
# from message_filters import Subscriber, ApproximateTimeSynchronizer
import threading
from concurrent.futures import ThreadPoolExecutor

import argparse
import utils.troubleshoot as troubleshoot

from perception_pipeline import PerceptionPipeline
from utils.camera_helpers import create_tf_matrix_from_euler, create_tf_matrix_from_msg

from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class RealTimePerceptionPipeline(PerceptionPipeline):
    def __init__(self, load_static_config=False):
        super().__init__()

        # load configs
        self.load_and_setup_pipeline_configs()

        if load_static_config:
            rospy.loginfo("loading static camera configs...")
            self.load_and_setup_static_camera_configs()

        # finish setup
        super().setup()


    def load_and_setup_pipeline_configs(self):
        self.perception_pipeline_config = rospy.get_param("perception_pipeline_config/", None)  
        self.scene_bounds = self.perception_pipeline_config['scene_bounds']
        self.cubic_size = self.perception_pipeline_config['voxel_props']['cubic_size']
        self.voxel_resolution = self.perception_pipeline_config['voxel_props']['voxel_resolution']


    def load_and_setup_static_camera_configs(self):
        self.static_camera_config = rospy.get_param("static_camera_config/", None)  

        def setup_cameras(static_camera_config):
            self.cameras = dict()
            for camera_name, camera_config in static_camera_config.items():
                self.cameras[camera_name] = dict()
                tf_matrix = create_tf_matrix_from_euler(camera_config['pose'])
                self.cameras[camera_name]['tf'] = tf_matrix
                rospy.loginfo(f"camera '{camera_name}' setup complete")
            self.camera_names = list(self.cameras.keys())
        setup_cameras(self.static_camera_config)


    def create_observation(self, msg:PointCloud2):
        obs = dict()
        for camera_name in self.camera_names:
            obs[camera_name] = dict()
            obs[camera_name]['pcd'] = msg
            obs[camera_name]['tf'] = self.cameras[camera_name]['tf']
        # add more to obs dict
        return obs

    def run_pipeline(self, msg:PointCloud2):
        try:
            ret = self.create_observation(msg)
            ret = self.run(ret)
            return self.make_pcd_msg(ret) 

        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))

    def make_pcd_msg(self, points_array):
        # Define header
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_1_link"  # Set the frame of reference

        points_list = points_array

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_msg = pc2.create_cloud(header, fields, points_list)
        return point_cloud_msg


class PerceptionNode:
    def __init__(self, args, max_threads=5):
        rospy.init_node('perception_node')
        # threading
        self.max_threads = max_threads
        self.executor = ThreadPoolExecutor(max_threads)
        self.lock = threading.Lock()

        # Subscribers
        rospy.Subscriber('/cameras/camera_1/depth/color/points', PointCloud2, self.callback)

        # Publishers
        self.publisher = rospy.Publisher('/primitives', PointCloud2, queue_size=10)

        # Setup Perception Pipeline            
        self.perception_pipeline = RealTimePerceptionPipeline(
            load_static_config=args.static
        )
        
    def callback(self, msg:PointCloud2):
        self.executor.submit(self.run_pipeline, msg)

    def publish_msg(self, msg:PointCloud2):
        self.publisher.publish(msg)

    def run_pipeline(self, msg:PointCloud2):
        msg = self.perception_pipeline.run_pipeline(msg)
        self.publish_msg(msg)

    def shutdown(self):
        self.executor.shutdown(wait=True)
        rospy.loginfo("Shutting down node.")

def main():
    parser = argparse.ArgumentParser(description="Configurable ROS Node")
    parser.add_argument('--static', action='store_true', help="Use static configuration instead of listening to a topic")
    args = parser.parse_args(rospy.myargv()[1:])  
    node = PerceptionNode(args, max_threads=5)
    return node

if __name__ == "__main__":
    try:
        node = main()
        rospy.spin()
    except rospy.ROSInterruptException:
        node.shutdown()
