#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os
import tkinter as tk
from tkinter import messagebox
import threading
import shutil

class TaskPlanner(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        
        self.declare_parameter('sync', False)
        self.sync_enabled = self.get_parameter('sync').get_parameter_value().bool_value
        
        self.global_path = '/tmp/planner_goals.txt'
        self.local_path = os.path.join(os.getcwd(), 'planner_goals.txt')
        self.buffer = []
        
        self.joint_state_pub = self.create_publisher(JointState, '/set_joint_states', 10)
        self.subscription = self.create_subscription(JointState, '/planner_input', self.planner_callback, 10)
        
        self.initialize_and_load()

        self.root = tk.Tk()
        self.root.title("Task Planner Goal Manager")
        self.setup_gui()

    def initialize_and_load(self):
        """
        Determines primary source of truth.
        Prioritizes local file only if sync is True and file exists.
        Otherwise, loads from global and synchronizes local if enabled.
        """
        source_path = self.global_path

        if self.sync_enabled and os.path.exists(self.local_path):
            source_path = self.local_path
            self.get_logger().info("Loading from local file and syncing to global.")
            if os.path.exists(source_path):
                shutil.copy2(self.local_path, self.global_path)
        else:
            self.get_logger().info("Loading from global file.")

        if os.path.exists(source_path):
            with open(source_path, 'r') as f:
                self.buffer = [line.strip() for line in f.readlines() if line.strip()]
        
        # Immediate sync of local if it doesn't exist but sync is enabled
        if self.sync_enabled and not os.path.exists(self.local_path):
            self.write_to_file(silent=True)

    def write_to_file(self, silent=False):
        """
        Saves buffer to file system.
        Always updates global. Updates local if sync is enabled.
        """
        try:
            # Update Global (Priority)
            with open(self.global_path, 'w') as f:
                for state in self.buffer:
                    f.write(f"{state}\n")
            
            # Update Local if enabled
            if self.sync_enabled:
                with open(self.local_path, 'w') as f:
                    for state in self.buffer:
                        f.write(f"{state}\n")
            
            if not silent:
                messagebox.showinfo("Success", "Goal files synchronized.")
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Write failed: {str(e)}")
            self.get_logger().error(f"Sync Failure: {str(e)}")

    def planner_callback(self, msg):
        formatted_state = " ".join([f"{pos:.4f}" for pos in msg.position])
        self.buffer.append(formatted_state)
        self.root.after(0, self.update_listbox_append, formatted_state)
        self.write_to_file(silent=True)

    def setup_gui(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10, padx=10)
        self.scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(frame, yscrollcommand=self.scrollbar.set, width=50, height=15)
        self.scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for item in self.buffer:
            self.listbox.insert(tk.END, item)

        action_frame = tk.Frame(self.root)
        action_frame.pack(pady=5)
        tk.Button(action_frame, text="Delete", command=self.delete_goal, fg="red").grid(row=0, column=0, padx=2)
        tk.Button(action_frame, text="Delete All", command=self.delete_all_goals, fg="white", bg="red").grid(row=0, column=1, padx=2)
        tk.Button(action_frame, text="Save (Sync Files)", command=lambda: self.write_to_file(silent=False), bg="green", fg="white").grid(row=0, column=2, padx=2)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Up", command=self.move_cursor_up).grid(row=0, column=0, padx=2)
        tk.Button(btn_frame, text="Down", command=self.move_cursor_down).grid(row=0, column=1, padx=2)
        tk.Button(btn_frame, text="Move Up", command=self.move_item_up).grid(row=0, column=2, padx=2)
        tk.Button(btn_frame, text="Move Down", command=self.move_item_down).grid(row=0, column=3, padx=2)

        sync_frame = tk.Frame(self.root)
        sync_frame.pack(pady=10)
        tk.Button(sync_frame, text="Apply Selected to Sliders", command=self.send_to_joint_manager, bg="#d1d1ff", height=2, width=40).pack()

    def update_listbox_append(self, state):
        self.listbox.insert(tk.END, state)
        self.listbox.see(tk.END)

    def move_cursor_up(self):
        idx = self.listbox.curselection()
        if idx and idx[0] > 0:
            self.listbox.selection_clear(idx[0]); self.listbox.selection_set(idx[0] - 1); self.listbox.activate(idx[0] - 1)

    def move_cursor_down(self):
        idx = self.listbox.curselection()
        if idx and idx[0] < self.listbox.size() - 1:
            self.listbox.selection_clear(idx[0]); self.listbox.selection_set(idx[0] + 1); self.listbox.activate(idx[0] + 1)

    def move_item_up(self):
        idx = self.listbox.curselection()
        if not idx or idx[0] == 0: return
        pos = idx[0]
        self.buffer[pos], self.buffer[pos-1] = self.buffer[pos-1], self.buffer[pos]
        val = self.listbox.get(pos); self.listbox.delete(pos); self.listbox.insert(pos - 1, val); self.listbox.selection_set(pos - 1)
        self.write_to_file(silent=True)

    def move_item_down(self):
        idx = self.listbox.curselection()
        if not idx or idx[0] == self.listbox.size() - 1: return
        pos = idx[0]
        self.buffer[pos], self.buffer[pos+1] = self.buffer[pos+1], self.buffer[pos]
        val = self.listbox.get(pos); self.listbox.delete(pos); self.listbox.insert(pos + 1, val); self.listbox.selection_set(pos + 1)
        self.write_to_file(silent=True)

    def delete_goal(self):
        idx = self.listbox.curselection()
        if idx:
            self.buffer.pop(idx[0]); self.listbox.delete(idx[0])
            self.write_to_file(silent=True)

    def delete_all_goals(self):
        if messagebox.askyesno("Confirm", "Delete all goals?"):
            self.buffer.clear(); self.listbox.delete(0, tk.END)
            self.write_to_file(silent=True)

    def send_to_joint_manager(self):
        idx = self.listbox.curselection()
        if not idx: return
        try:
            selected_str = self.listbox.get(idx[0])
            positions = [float(val) for val in selected_str.split()]
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [f"panda_joint{i+1}" for i in range(len(positions))]
            msg.position = positions
            self.joint_state_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Publish error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlanner()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    try:
        node.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()