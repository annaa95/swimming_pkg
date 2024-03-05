import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from hb40_commons.msg import JointCommand

import sys, os

from .swimming_mab_robot.env.hopf_network_gui import HopfNetwork

class SwimmingNode(Node):

    def __init__(self):
        super().__init__('swimming_node')
        self.publisher_ = self.create_publisher(JointCommand, '/hb40/joint_command', 10)
        self.subscriber_ = self.create_subscription(JointState, '/hb40/joint_states', self.subscriber_callback, 10)
        timer_period = 0.005  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
    
    def subscriber_callback(self,msg_read ) :
        # read the joint state 
        self.get_logger().info("Reading:"+str(msg_read.position))
        self.q_read_ = msg_read.position

    def timer_callback(self):
        # cpg dynamics
        # foot trajectory

        # inverse kinematics
        msg = JointCommand()
        msg.name = ["fr_j0","fr_j1","fr_j2", 
                "fl_j0","fl_j1", "fl_j2", 
                "rl_j0", "rl_j1", "rl_j2", 
                "rr_j0", "rr_j1", "rr_j2"]
        msg.kp= [500.0]*12
        msg.kd= [10.0]*12
        msg.t_pos = [30.0]*12
        msg.t_vel = [50.0]*12

        self.publisher_.publish(msg)



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = SwimmingNode()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()