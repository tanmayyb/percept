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
        
        self.subscription = self.create_subscription(JointState, '/joint_states', self.listener_callback, 10)
        self.publisher = self.create_publisher(JointState, '/set_joint_states', 10)
        self.planner_publisher = self.create_publisher(JointState, '/planner_input', 10)
            
        self.current_positions = []
        self.joint_names = []
        
        # Path definitions
        self.local_path = os.path.join(os.getcwd(), 'start_configuration.yaml')
        try:
            package_share_dir = get_package_share_directory('experiments')
            self.share_dir = os.path.join(package_share_dir, 'manipulator')
            self.share_path = os.path.join(self.share_dir, 'start_configuration.yaml')
        except Exception:
            self.share_path = None

        # Execute Startup Sync
        self.sync_on_startup()

    def listener_callback(self, msg):
        self.current_positions = list(msg.position)
        if not self.joint_names:
            self.joint_names = list(msg.name)

    def _read_yaml(self, path):
        """Standardized yaml loader."""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                try:
                    return yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    return {}
        return {}

    def sync_on_startup(self):
        """Merges files on launch to ensure local and global consistency."""
        global_data = self._read_yaml(self.share_path)
        local_data = self._read_yaml(self.local_path)

        # Merge logic: Start with Global, Overwrite with Local
        merged_data = global_data.copy()
        merged_data.update(local_data)

        if not merged_data:
            print("Startup: No configuration data found to sync.")
            return

        # Write merged result to Local (Primary)
        with open(self.local_path, 'w') as f:
            yaml.dump(merged_data, f, default_flow_style=False)
        
        # Sync merged result to Global (Scratch)
        if self.share_path:
            os.makedirs(self.share_dir, exist_ok=True)
            with open(self.share_path, 'w') as f:
                yaml.dump(merged_data, f, default_flow_style=False)
        
        print("Startup Sync Complete: Local and Global files are now identical.")

    def save_configuration(self, key):
        """Updates primary local config and syncs it back to global scratch."""
        if not self.current_positions:
            messagebox.showwarning("Warning", "No joint data to save.")
            return

        # Load latest state from local (already synced with global on startup)
        data = self._read_yaml(self.local_path)
        
        # Update specific key
        data[key] = [round(float(pos), 4) for pos in self.current_positions]

        # Save to both locations
        with open(self.local_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        if self.share_path:
            with open(self.share_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        
        print(f"Saved and Synced: {key}")

    def publish_configuration(self, key):
        """Loads from local primary file."""
        data = self._read_yaml(self.local_path)
        
        if key not in data:
            messagebox.showwarning("Warning", f"Key '{key}' not found.")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = data[key]
        self.publisher.publish(msg)

    def send_to_planner(self):
        if not self.current_positions: return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_positions
        self.planner_publisher.publish(msg)

    def copy_to_clipboard(self, window):
        if not self.current_positions: return
        formatted_text = "\n".join([f"- {pos:.4f}" for pos in self.current_positions])
        window.clipboard_clear()
        window.clipboard_append(formatted_text)
        window.update()

def main():
    rclpy.init()
    node = JointStateGui()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    window = tk.Tk()
    window.title("ROS2 Joint Manager (Synced)")
    window.geometry("400x580")

    tk.Label(window, text="Files Synced on Launch", font=('Arial', 9, 'bold'), fg="green").pack(pady=10)
    
    tk.Button(window, text="Copy Current State", command=lambda: node.copy_to_clipboard(window), height=2, width=30, bg="#f0f0f0").pack(pady=5)
    tk.Button(window, text="Send to Task Planner", command=node.send_to_planner, height=2, width=30, bg="#ffd1d1").pack(pady=5)
    
    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=15, pady=10)
    
    tk.Button(window, text="Save current as Start Goal", command=lambda: node.save_configuration("start_configuration"), height=2, width=30, bg="#d1ffd1").pack(pady=5)
    tk.Button(window, text="Save current as End Goal", command=lambda: node.save_configuration("goal_configuration"), height=2, width=30, bg="#d1ffd1").pack(pady=5)

    tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=15, pady=10)
    
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