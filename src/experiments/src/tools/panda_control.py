#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import panda_py
from panda_py import controllers
import numpy as np
import threading
import time

# Joint,Range (Radians)
# 1,-2.8973 to 2.8973
# 2,-1.7628 to 1.7628
# 3,-2.8973 to 2.8973
# 4,-3.0718 to -0.0698
# 5,-2.8973 to 2.8973
# 6,-0.0175 to 3.7525
# 7,-2.8973 to 2.8973

# Index,Component,Type,Units,Physical Meaning
# 0,Kx​,Translational,N/m,Resistance to being pushed along the X-axis.
# 1,Ky​,Translational,N/m,Resistance to being pushed along the Y-axis.
# 2,Kz​,Translational,N/m,Resistance to being pushed along the Z-axis.
# 3,"Krot,x​",Rotational,Nm/rad,Resistance to twisting around the X-axis (Roll).
# 4,"Krot,y​",Rotational,Nm/rad,Resistance to twisting around the Y-axis (Pitch).
# 5,"Krot,z​",Rotational,Nm/rad,Resistance to twisting around the Z-axis (Yaw).

class PandaCartesianController(Node):
    def __init__(self):
        super().__init__('panda_cartesian_controller')
        
        # Parameters
        self.declare_parameter('robot_ip', '172.22.2.4')
        self.declare_parameter('impedance_stiffness', [600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
        self.declare_parameter('damping_ratio', 1.0)
        self.declare_parameter('nullspace_stiffness', 0.01)
        self.declare_parameter('filter_coeff', 1.0)
        self.declare_parameter('state_publish_rate', 30.0) # Hz

        # Initialize Robot
        ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.panda = panda_py.Panda(ip)
        
        # Initialize Controller
        stiffness_diag = self.get_parameter('impedance_stiffness').get_parameter_value().double_array_value
        self.ctrl = controllers.CartesianImpedance(
            impedance=np.diag(stiffness_diag),
            damping_ratio=self.get_parameter('damping_ratio').value,
            nullspace_stiffness=self.get_parameter('nullspace_stiffness').value,
            filter_coeff=self.get_parameter('filter_coeff').value
        )

        self.current_pose_target = None
        self.lock = threading.Lock()
        
        # Pub/Sub
        self.pose_sub = self.create_subscription(PoseStamped, '/manipulator/pose', self.pose_callback, 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Control and State Threads
        self.control_thread = threading.Thread(target=self.run_control_loop, daemon=True)
        self.state_thread = threading.Thread(target=self.publish_joint_states, daemon=True)
        
        self.control_thread.start()
        self.state_thread.start()

    def pose_callback(self, msg: PoseStamped):
        with self.lock:
            pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            # Mapping ROS (x,y,z,w) to panda_py (w,x,y,z)
            # ori = np.array([msg.pose.orientation.w, msg.pose.orientation.x, 
            #                 msg.pose.orientation.y, msg.pose.orientation.z])
            ori = np.array([
              msg.pose.orientation.x, 
              msg.pose.orientation.y, 
              msg.pose.orientation.z,
              msg.pose.orientation.w, 
            ])
            self.current_pose_target = (pos, ori)

    def publish_joint_states(self):
        rate = self.get_parameter('state_publish_rate').value
        sleep_time = 1.0 / rate
        joint_names = [f"panda_joint{i+1}" for i in range(7)]
        
        while rclpy.ok():
            # Retrieve real-time state from hardware
            state = self.panda.get_state()
            
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = state.q
            msg.velocity = state.dq
            msg.effort = state.tau_J
            
            self.joint_pub.publish(msg)
            time.sleep(sleep_time)

    def run_control_loop(self):
        q_null = np.array([0, -0.78539816, 0, -2.35619449, 0, 1.57079633, 0.78539816])
        self.panda.start_controller(self.ctrl)
        
        while rclpy.ok():
            with self.lock:
                if self.current_pose_target is not None:
                    pos, ori = self.current_pose_target
                    self.ctrl.set_control(pos, ori, q_null)
            time.sleep(0.001) # Maintains loop availability
            
    def stop(self):
        self.panda.stop_controller()

def main(args=None):
    rclpy.init(args=args)
    node = PandaCartesianController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()