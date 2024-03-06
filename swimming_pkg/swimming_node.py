import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from hb40_commons.msg import JointCommand
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import sys
import os
import numpy as np
import pickle

from .swimming_mab_robot.env.hopf_network_gui import HopfNetwork

HIP_LINK_LENGTH = 0.064
THIGH_LINK_LENGTH = 0.2
CALF_LINK_LENGTH = 0.2

csv_config = {'gait': np.array([0., -1.57079633,  3.14159265,  1.57079633]),
              'omega_swing': 0.1*7.5398223686155035,
              'omega_stance': 0.1*21.991148575128552,
              'horizontal_offset': -0.01,
              'vertical_offset': -0.18,
              'des_step_len': 0.26,
              'step_height': 0.08,
              'belly_length': 0.010400000000000001,
              'belly_curvature': 0.49,
              'compression_ratio': 0.23,
              'inclination': 0.0,
              'alpha': 5}

CTRL_TIMESTEP = 1/200
cpg = HopfNetwork(**csv_config, time_step=CTRL_TIMESTEP,)


class SwimmingNode(Node):

    def __init__(self):
        super().__init__('swimming_node')
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(
            JointCommand, '/hb40/joint_command', qos_pub)
        self.publisher_debug_ = self.create_publisher(
            JointCommand, '/hb40/debug/joint_command', qos_pub)
        self.subscriber_ = self.create_subscription(
            JointState, '/hb40/joint_states', self.subscriber_callback, qos_sub)
        self.timer_period = CTRL_TIMESTEP  # seconds
        self.kp_ = 5.0
        self.kd_ = .1
        self.counter_ = 0
        self.trj_ = self.get_dummy_trajectory()
        self.old_qdes_ = np.zeros(12)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def get_dummy_trajectory(self):
        # create a dummy trajectory in cartesian space
        xyz = np.zeros((1000, 3))
        for t in range(1000):
            xyz[t] = np.array([0.1+0.1*np.cos(2*np.pi*t/1000.),
                              HIP_LINK_LENGTH, -0.15+0.1*np.sin(2*np.pi*t/1000.)])
        return xyz

    def ComputeInverseKinematicsHipPlane(self, legID, xyz_coord):
        """ Get joint angles for leg legID with desired xyz position in leg frame. 
        Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;
        From SpotMicro: 
        https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
        """
        # rename
        l0 = HIP_LINK_LENGTH
        l1 = THIGH_LINK_LENGTH
        l2 = CALF_LINK_LENGTH

        # coords
        x = xyz_coord[0]
        y = xyz_coord[1]
        z = xyz_coord[2]

        q0 = 0.
        L = np.sqrt(x**2+z**2)
        if legID == 0 or legID == 3:
            legsign = -1
        else:
            legsign = 1

        q2 = legsign*np.arccos((l1**2+l2**2-L**2)/(2*l1*l2))
        alpha = np.arctan2(-z, -x)
        gamma = np.arcsin(l2*np.sin(np.abs(q2))/L)
        q1 = -legsign*(alpha-gamma)

        q = np.array([q0, q1, q2])

        return list(q)

    def subscriber_callback(self, msg_read):
        # read the joint state
        # self.get_logger().info("Reading:"+str(msg_read.position))
        self.q_read_ = msg_read.position

    def timer_callback(self):
        # loop at 100Hz, counter 1 -> 0.01 s
        # cpg dynamics
        xs, zs = cpg.update()

        # foot trajectory
        next_xyz = self.trj_[self.counter_%len(self.trj_)][:]

        # inverse kinematics
        q_des = []
        for leg in range(4):
            next_xyz = np.array([xs[leg], HIP_LINK_LENGTH, zs[leg]])
            q_des.extend(self.ComputeInverseKinematicsHipPlane(leg, next_xyz))
        # write command to be published
        msg = JointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source_node = "Swimming_node"
        msg.name = ["fr_j0", "fr_j1", "fr_j2",
                    "fl_j0", "fl_j1", "fl_j2",
                    "rl_j0", "rl_j1", "rl_j2",
                    "rr_j0", "rr_j1", "rr_j2", "sp_j0"]

        # msg.kp = [self.kp_]*12+[0.]
        msg.kp = [self.kp_ for x in range(13)]
        # msg.kd = [self.kd_]*12+[0.]
        msg.kd = [self.kd_ for x in range(13)]
        """
        hip_angle = 0.0
        thigh_angle = np.pi/6*np.sin(2*np.pi*self.counter_*self.timer_period * 0.5)
        calf_angle = np.pi/2

        msg.t_pos = [hip_angle, thigh_angle, -calf_angle,
                     hip_angle, thigh_angle, calf_angle,
                     hip_angle, thigh_angle, calf_angle,
                     hip_angle, thigh_angle, -calf_angle, 0.0]
        """

        msg.t_vel = [0.0 for x in range(13)]
        msg.t_trq = [0.0 for x in range(13)]
        msg.t_pos = q_des+[0.0]
        msg.t_vel = list((np.array(q_des)-self.old_qdes_)/CTRL_TIMESTEP)+[0.0]
        self.counter_ += 1
        self.old_qdes_ = np.array(q_des)

        self.publisher_.publish(msg)
        self.publisher_debug_.publish(msg)


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
