#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from percept_interfaces.srv import SetGoal

class ManipulatorClient(Node):
    def __init__(self):
        super().__init__('manipulator_client_py')
        self.client = self.create_client(SetGoal, '/manipulator/set_goal_callback')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
            
        self.request = SetGoal.Request()

    def send_request(self, positions):
        self.request.joint_positions = positions
        self.future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    client_node = ManipulatorClient()
    
    # Arbitrary data
    # arbitrary_positions = [0.5, -1.2, 0.75, 0.0, 1.57, -0.4, -0.4]
    arbitrary_positions = [
      0.0000,
      0.0095,
      0.0000,
      -1.2544,
      0.0151,
      2.5299,
      0.6415,
    ]
    
    
    response = client_node.send_request(arbitrary_positions)
    
    if response is not None:
        client_node.get_logger().info(f'Result: {response.success}')
    else:
        client_node.get_logger().error('Service call failed')

    client_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()