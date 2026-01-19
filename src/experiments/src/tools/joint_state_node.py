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
        
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        
        self.publisher = self.create_publisher(
            JointState, 
            '/set_joint_states', 
            10)

        self.planner_publisher = self.create_publisher(
            JointState,
            '/planner_input',
            10)
            
        self.current_positions = []
        self.joint_names = []
        
        try:
            package_share_dir = get_package_share_directory('experiments')
            self.config_dir = os.path.join(package_share_dir, 'manipulator')
            self.config_path = os.path.join(self.config_dir, 'start_configuration.yaml')
        except Exception:
            self.config_path = None

    def listener_callback(self, msg):
        self.current_positions = list(msg.position)
        if not self.joint_names:
            self.joint_names = list(msg.name)

    def send_to_planner(self):
        if not self.current_positions:
            messagebox.showwarning("Warning", "No joint state data received yet.")
            return
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_positions
        self.planner_publisher.publish(msg)
        print("Joint state sent to TaskPlanner.")

    def save_configuration(self, key):
        if not self.current_positions or not self.config_path:
            messagebox.showwarning("Warning", "Data or path unavailable.")
            return

        os.makedirs(self.config_dir, exist_ok=True)
        data = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    data = {}

        data[key] = [round(float(pos), 4) for pos in self.current_positions]

        with open(self.config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Saved {key} to {self.config_path}")

    def publish_configuration(self, key):
        if not os.path.exists(self.config_path) or not self.joint_names:
            messagebox.showwarning("Warning", "Configuration file or joint names not found.")
            return

        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if key not in data:
            messagebox.showwarning("Warning", f"Key '{key}' not found in YAML.")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = data[key]
        self.publisher.publish(msg)

    def copy_to_clipboard(self, window):
        if not self.current_positions:
            messagebox.showwarning("Warning", "No joint state data received yet.")
            return

        formatted_text = "\n".join([f"- {pos:.4f}" for pos in self.current_positions])
        window.clipboard_clear()
        window.clipboard_append(formatted_text)
        window.update()
        print("Configuration copied to clipboard.")

def main():
    rclpy.init()
    node = JointStateGui()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    window = tk.Tk()
    window.title("ROS2 Joint State Manager")
    window.geometry("400x580")

    tk.Label(window, text="Joint State Monitor active", font=('Arial', 10, 'bold')).pack(pady=10)

    tk.Label(window, text="Clipboard & Planning", font=('Arial', 9, 'italic')).pack()
    tk.Button(
        window, 
        text="Copy Current State", 
        command=lambda: node.copy_to_clipboard(window), 
        height=2, width=30, bg="#f0f0f0"
    ).pack(pady=5)

    tk.Button(
        window, 
        text="Send to Task Planner", 
        command=node.send_to_planner, 
        height=2, width=30, bg="#ffd1d1"
    ).pack(pady=5)

    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=15, pady=10)

    tk.Label(window, text="File Persistence (YAML)", font=('Arial', 9, 'italic')).pack()
    tk.Button(
        window, 
        text="Save current as Start Goal", 
        command=lambda: node.save_configuration("start_configuration"), 
        height=2, width=30, bg="#d1ffd1"
    ).pack(pady=5)
    
    tk.Button(
        window, 
        text="Save current as End Goal", 
        command=lambda: node.save_configuration("goal_configuration"), 
        height=2, width=30, bg="#d1ffd1"
    ).pack(pady=5)

    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=15, pady=10)

    tk.Label(window, text="Remote GUI Control", font=('Arial', 9, 'italic')).pack()
    tk.Button(
        window, 
        text="Apply Start Goal to Sliders", 
        command=lambda: node.publish_configuration("start_configuration"), 
        height=2, width=30, bg="#d1d1ff"
    ).pack(pady=5)
    
    tk.Button(
        window, 
        text="Apply End Goal to Sliders", 
        command=lambda: node.publish_configuration("goal_configuration"), 
        height=2, width=30, bg="#d1d1ff"
    ).pack(pady=5)

    try:
        window.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()