#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tkinter as tk
from tkinter import messagebox
import threading
import os
import yaml
from ament_index_python.packages import get_package_share_directory

class JointStateGui(Node):
    def __init__(self):
        super().__init__('joint_state_manager_node')
        
        # ROS2 Parameter for sync control
        self.declare_parameter('use_local_sync', False)
        self.use_local_sync = self.get_parameter('use_local_sync').get_parameter_value().bool_value
        
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        
        self.publisher = self.create_publisher(JointState, '/set_joint_states', 10)
        self.planner_publisher = self.create_publisher(JointState, '/planner_input', 10)
            
        self.current_positions = []
        self.joint_names = []
        
        # Path definitions
        self.local_config_path = os.path.join(os.getcwd(), 'start_configuration.yaml')
        try:
            package_share_dir = get_package_share_directory('experiments')
            self.config_dir = os.path.join(package_share_dir, 'manipulator')
            self.share_config_path = os.path.join(self.config_dir, 'start_configuration.yaml')
        except Exception:
            self.share_config_path = None

    def listener_callback(self, msg):
        self.current_positions = list(msg.position)
        if not self.joint_names:
            self.joint_names = list(msg.name)

    def save_configuration(self, key):
        """Reads from share, updates key, writes to both share and local."""
        if not self.current_positions or not self.share_config_path:
            messagebox.showwarning("Warning", "Joint data or Share path unavailable.")
            return

        # 1. Read existing data from the share directory to ensure we have all keys
        data = {}
        if os.path.exists(self.share_config_path):
            with open(self.share_config_path, 'r') as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    data = {}

        # 2. Update the specific key with new positions
        data[key] = [round(float(pos), 4) for pos in self.current_positions]

        # 3. Write to Share Directory (Main Source)
        os.makedirs(self.config_dir, exist_ok=True)
        with open(self.share_config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Updated Share Config: {self.share_config_path}")

        # 4. Sync to Local Directory
        if self.use_local_sync:
            with open(self.local_config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"Synced to Local Config: {self.local_config_path}")

    def publish_configuration(self, key):
        """Always loads from the Share directory."""
        if not self.share_config_path or not os.path.exists(self.share_config_path):
            messagebox.showwarning("Warning", "Share directory configuration file not found.")
            return

        with open(self.share_config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if not data or key not in data:
            messagebox.showwarning("Warning", f"Key '{key}' not found in {self.share_config_path}.")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = data[key]
        self.publisher.publish(msg)
        print(f"Published {key} from share directory.")

    def send_to_planner(self):
        if not self.current_positions:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_positions
        self.planner_publisher.publish(msg)

    def copy_to_clipboard(self, window):
        if not self.current_positions:
            return
        formatted_text = "\n".join([f"- {pos:.4f}" for pos in self.current_positions])
        window.clipboard_clear()
        window.clipboard_append(formatted_text)
        window.update()

def main():
    rclpy.init()
    node = JointStateGui()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    window = tk.Tk()
    window.title("ROS2 Joint State Manager")
    window.geometry("400x580")

    tk.Label(window, text="Joint State Monitor active", font=('Arial', 10, 'bold')).pack(pady=10)
    tk.Button(window, text="Copy Current State", command=lambda: node.copy_to_clipboard(window), height=2, width=30, bg="#f0f0f0").pack(pady=5)
    tk.Button(window, text="Send to Task Planner", command=node.send_to_planner, height=2, width=30, bg="#ffd1d1").pack(pady=5)
    
    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=15, pady=10)
    
    # Save Buttons (Syncs Share and Local)
    tk.Button(window, text="Save current as Start Goal", command=lambda: node.save_configuration("start_configuration"), height=2, width=30, bg="#d1ffd1").pack(pady=5)
    tk.Button(window, text="Save current as End Goal", command=lambda: node.save_configuration("goal_configuration"), height=2, width=30, bg="#d1ffd1").pack(pady=5)

    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=15, pady=10)
    
    # Load Buttons (Reads Share Only)
    tk.Button(window, text="Apply Start Goal to Sliders", command=lambda: node.publish_configuration("start_configuration"), height=2, width=30, bg="#d1d1ff").pack(pady=5)
    tk.Button(window, text="Apply End Goal to Sliders", command=lambda: node.publish_configuration("goal_configuration"), height=2, width=30, bg="#d1d1ff").pack(pady=5)

    try:
        window.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()