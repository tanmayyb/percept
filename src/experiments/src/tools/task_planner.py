#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os
import tkinter as tk
from tkinter import messagebox
import threading

class TaskPlanner(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.file_path = '/tmp/planner_goals.txt'
        self.buffer = []
        
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/set_joint_states',
            10)

        self.subscription = self.create_subscription(
            JointState,
            '/planner_input',
            self.planner_callback,
            10)
        
        self.load_existing_goals()

        self.root = tk.Tk()
        self.root.title("Task Planner Goal Manager")
        self.setup_gui()

    def setup_gui(self):
        # Listbox and Scrollbar
        frame = tk.Frame(self.root)
        frame.pack(pady=10, padx=10)

        self.scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(frame, yscrollcommand=self.scrollbar.set, width=50, height=15)
        self.scrollbar.config(command=self.listbox.yview)
        
        self.listbox.pack(side=tk.LEFT)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for item in self.buffer:
            self.listbox.insert(tk.END, item)

        # Action Frame (Delete and Save) - Now placed above Navigation
        action_frame = tk.Frame(self.root)
        action_frame.pack(pady=5)
        
        tk.Button(action_frame, text="Delete", command=self.delete_goal, fg="red").grid(row=0, column=0, padx=2)
        tk.Button(action_frame, text="Delete All", command=self.delete_all_goals, fg="white", bg="red").grid(row=0, column=1, padx=2)
        tk.Button(action_frame, text="Save to File", command=self.write_to_file, bg="green", fg="white").grid(row=0, column=2, padx=2)

        # Navigation and Reordering Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Up", command=self.move_cursor_up).grid(row=0, column=0, padx=2)
        tk.Button(btn_frame, text="Down", command=self.move_cursor_down).grid(row=0, column=1, padx=2)
        tk.Button(btn_frame, text="Move Up", command=self.move_item_up).grid(row=0, column=2, padx=2)
        tk.Button(btn_frame, text="Move Down", command=self.move_item_down).grid(row=0, column=3, padx=2)

        # Force Set Button
        sync_frame = tk.Frame(self.root)
        sync_frame.pack(pady=10)
        tk.Button(sync_frame, text="Apply Selected to Sliders", 
                  command=self.send_to_joint_manager, 
                  bg="#d1d1ff", height=2, width=40).pack()

    def load_existing_goals(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.buffer = [line.strip() for line in f.readlines() if line.strip()]
            self.get_logger().info(f"Loaded {len(self.buffer)} existing goals.")
        else:
            self.buffer = []

    def planner_callback(self, msg):
        formatted_state = " ".join([f"{pos:.4f}" for pos in msg.position])
        self.buffer.append(formatted_state)
        self.root.after(0, self.update_listbox_append, formatted_state)

    def send_to_joint_manager(self):
        idx = self.listbox.curselection()
        if not idx:
            messagebox.showwarning("Warning", "Please select a joint state.")
            return

        try:
            selected_str = self.listbox.get(idx[0])
            positions = [float(val) for val in selected_str.split()]
            joint_names = [f"panda_joint{i+1}" for i in range(len(positions))]
            
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = positions
            
            self.joint_state_pub.publish(msg)
            self.get_logger().info(f"Published state to /set_joint_states.")
        except Exception as e:
            self.get_logger().error(f"Error publishing state: {str(e)}")

    def update_listbox_append(self, state):
        self.listbox.insert(tk.END, state)
        self.listbox.see(tk.END)

    def move_cursor_up(self):
        idx = self.listbox.curselection()
        if idx and idx[0] > 0:
            self.listbox.selection_clear(idx[0])
            self.listbox.selection_set(idx[0] - 1)
            self.listbox.activate(idx[0] - 1)

    def move_cursor_down(self):
        idx = self.listbox.curselection()
        if idx and idx[0] < self.listbox.size() - 1:
            self.listbox.selection_clear(idx[0])
            self.listbox.selection_set(idx[0] + 1)
            self.listbox.activate(idx[0] + 1)

    def move_item_up(self):
        idx = self.listbox.curselection()
        if not idx or idx[0] == 0: return
        pos = idx[0]
        self.buffer[pos], self.buffer[pos-1] = self.buffer[pos-1], self.buffer[pos]
        val = self.listbox.get(pos)
        self.listbox.delete(pos)
        self.listbox.insert(pos - 1, val)
        self.listbox.selection_set(pos - 1)

    def move_item_down(self):
        idx = self.listbox.curselection()
        if not idx or idx[0] == self.listbox.size() - 1: return
        pos = idx[0]
        self.buffer[pos], self.buffer[pos+1] = self.buffer[pos+1], self.buffer[pos]
        val = self.listbox.get(pos)
        self.listbox.delete(pos)
        self.listbox.insert(pos + 1, val)
        self.listbox.selection_set(pos + 1)

    def delete_goal(self):
        idx = self.listbox.curselection()
        if idx:
            self.buffer.pop(idx[0])
            self.listbox.delete(idx[0])

    def delete_all_goals(self):
        if messagebox.askyesno("Confirm", "Delete all goals?"):
            self.buffer.clear()
            self.listbox.delete(0, tk.END)

    def write_to_file(self):
        try:
            with open(self.file_path, 'w') as f:
                for state in self.buffer:
                    f.write(f"{state}\n")
            messagebox.showinfo("Success", "File saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

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