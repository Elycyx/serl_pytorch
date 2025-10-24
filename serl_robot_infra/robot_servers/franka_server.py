"""
Franka robot server implementation without ROS dependencies.
This uses the Franka Python SDK (libfranka-python) for direct robot control.

Installation:
    pip install libfranka  # Or use the official Franka Python bindings

Note: This is a template implementation. You'll need to install and configure
the actual Franka Python SDK based on your robot's version and setup.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import time
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from serl_robot_infra.robot_servers.base_robot_server import BaseRobotServer

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "Robotiq", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)


class FrankaServer(BaseRobotServer):
    """
    Franka robot control server using direct SDK communication (no ROS).
    
    This is a template implementation that should be completed with actual
    Franka SDK calls based on your specific robot configuration.
    """

    def __init__(self, robot_ip: str, gripper_type: str = "Robotiq", reset_joint_target: list = None, **kwargs):
        """
        Initialize Franka robot server.
        
        Args:
            robot_ip: IP address of Franka robot controller
            gripper_type: Type of gripper ('Robotiq', 'Franka', or None)
            reset_joint_target: Target joint angles for reset position
            **kwargs: Additional configuration
        """
        self.robot_ip = robot_ip
        self.gripper_type = gripper_type
        self.reset_joint_target = reset_joint_target or [0, 0, 0, -1.9, 0, 2, 0]
        
        # Robot state
        self._current_joint_positions = None
        self._current_joint_velocities = None
        self._current_cartesian_pose = None
        self._jacobian = None
        self._is_connected = False
        
        # TODO: Initialize actual Franka robot connection here
        # Example (pseudo-code):
        # from franka import Robot
        # self.robot = Robot(robot_ip)
        
        print(f"[FrankaServer] Initialized for robot at {robot_ip}")
        print(f"[FrankaServer] Gripper type: {gripper_type}")
        print("[FrankaServer] NOTE: This is a template implementation.")
        print("[FrankaServer] Please implement actual Franka SDK calls based on your robot configuration.")
    
    def connect(self) -> bool:
        """
        Connect to the Franka robot.
        
        Returns:
            True if connection successful
        """
        try:
            # TODO: Implement actual connection logic using Franka SDK
            # Example (pseudo-code):
            # self.robot.connect()
            # self._is_connected = self.robot.is_connected()
            
            print("[FrankaServer] Connect called - implement actual SDK connection")
            self._is_connected = True  # Placeholder
            return self._is_connected
        except Exception as e:
            print(f"[FrankaServer] Connection failed: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Franka robot.
        
        Returns:
            True if disconnection successful
        """
        try:
            # TODO: Implement actual disconnection logic
            # Example (pseudo-code):
            # self.robot.disconnect()
            
            print("[FrankaServer] Disconnect called")
            self._is_connected = False
            return True
        except Exception as e:
            print(f"[FrankaServer] Disconnection failed: {e}")
            return False
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current robot state.
        
        Returns:
            Dictionary with joint positions, velocities, cartesian pose, etc.
        """
        # TODO: Implement actual state reading from Franka SDK
        # Example (pseudo-code):
        # state = self.robot.read_once()
        # return {
        #     'joint_positions': np.array(state.q),
        #     'joint_velocities': np.array(state.dq),
        #     'joint_torques': np.array(state.tau_J),
        #     'cartesian_position': state.O_T_EE[:3, 3],
        #     'cartesian_orientation': ...,
        # }
        
        # Placeholder return
        return {
            'joint_positions': self._current_joint_positions or np.zeros(7),
            'joint_velocities': self._current_joint_velocities or np.zeros(7),
            'joint_torques': np.zeros(7),
            'cartesian_position': np.zeros(3),
            'cartesian_orientation': np.array([0, 0, 0, 1]),  # quaternion
        }
    
    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move to target joint positions.
        
        Args:
            positions: Target joint positions (7 values for Franka)
            velocity: Maximum velocity
            acceleration: Maximum acceleration
            blocking: Whether to wait for completion
            
        Returns:
            True if successful
        """
        try:
            # TODO: Implement actual joint motion using Franka SDK
            # Example (pseudo-code):
            # motion_generator = JointMotion(positions, velocity, acceleration)
            # self.robot.control(motion_generator)
            
            print(f"[FrankaServer] Moving to joint positions: {positions}")
            self._current_joint_positions = positions
            return True
        except Exception as e:
            print(f"[FrankaServer] Joint motion failed: {e}")
            return False
    
    def move_to_cartesian_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move to target Cartesian pose.
        
        Args:
            position: Target position [x, y, z]
            orientation: Target orientation (quaternion [qx, qy, qz, qw])
            velocity: Maximum velocity
            acceleration: Maximum acceleration
            blocking: Whether to wait for completion
            
        Returns:
            True if successful
        """
        try:
            # TODO: Implement actual Cartesian motion using Franka SDK
            # Example (pseudo-code):
            # pose = create_pose_matrix(position, orientation)
            # motion_generator = CartesianMotion(pose, velocity, acceleration)
            # self.robot.control(motion_generator)
            
            print(f"[FrankaServer] Moving to Cartesian pose: pos={position}, ori={orientation}")
            return True
        except Exception as e:
            print(f"[FrankaServer] Cartesian motion failed: {e}")
            return False
    
    def send_joint_command(
        self,
        command: np.ndarray,
        command_type: str = 'position',
    ) -> bool:
        """
        Send low-level joint command.
        
        Args:
            command: Joint command values
            command_type: 'position', 'velocity', or 'torque'
            
        Returns:
            True if successful
        """
        try:
            # TODO: Implement actual command sending using Franka SDK
            # Example (pseudo-code):
            # if command_type == 'position':
            #     self.robot.set_joint_positions(command)
            # elif command_type == 'torque':
            #     self.robot.set_joint_torques(command)
            
            print(f"[FrankaServer] Sending {command_type} command: {command}")
            return True
        except Exception as e:
            print(f"[FrankaServer] Command failed: {e}")
            return False
    
    def send_cartesian_command(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Send Cartesian space command (for impedance control).
        
        Args:
            position: Target position
            orientation: Target orientation (if None, keep current)
            
        Returns:
            True if successful
        """
        try:
            # TODO: Implement Cartesian impedance control command
            # This typically involves setting equilibrium pose for impedance controller
            
            print(f"[FrankaServer] Sending Cartesian command: {position}")
            return True
        except Exception as e:
            print(f"[FrankaServer] Cartesian command failed: {e}")
            return False
    
    def reset(self) -> bool:
        """
        Reset robot to home position.
        
        Returns:
            True if successful
        """
        try:
            # Move to reset joint configuration
            return self.move_to_joint_positions(
                np.array(self.reset_joint_target),
                blocking=True
            )
        except Exception as e:
            print(f"[FrankaServer] Reset failed: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Emergency stop.
        
        Returns:
            True if successful
        """
        try:
            # TODO: Implement emergency stop using Franka SDK
            # Example (pseudo-code):
            # self.robot.stop()
            
            print("[FrankaServer] Emergency stop called")
            return True
        except Exception as e:
            print(f"[FrankaServer] Stop failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to robot.
        
        Returns:
            True if connected
        """
        return self._is_connected
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Franka joint limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        # Franka Panda joint limits (radians)
        lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        return lower, upper
    
    def get_workspace_limits(self) -> Dict[str, np.ndarray]:
        """
        Get approximate workspace limits for Franka.
        
        Returns:
            Dictionary with position min/max
        """
        return {
            'position_min': np.array([0.3, -0.5, 0.0]),
            'position_max': np.array([0.8, 0.5, 0.6]),
        }
    
    def get_jacobian(self) -> Optional[np.ndarray]:
        """
        Get current Jacobian matrix.
        
        Returns:
            6x7 Jacobian matrix
        """
        # TODO: Implement Jacobian retrieval from Franka SDK
        # The Franka provides this directly from the robot state
        return self._jacobian
    
    def error_recovery(self) -> bool:
        """
        Perform error recovery (Franka-specific).
        
        Returns:
            True if successful
        """
        try:
            # TODO: Implement error recovery using Franka SDK
            # Example (pseudo-code):
            # self.robot.automatic_error_recovery()
            
            print("[FrankaServer] Error recovery called")
            return True
        except Exception as e:
            print(f"[FrankaServer] Error recovery failed: {e}")
            return False


def create_app(franka_server: FrankaServer):
    """
    Create Flask app for HTTP API to robot server.
    
    Args:
        franka_server: Franka server instance
        
    Returns:
        Flask app
    """
    app = Flask(__name__)
    
    @app.route('/get_state', methods=['GET'])
    def get_state():
        state = franka_server.get_state()
        # Convert numpy arrays to lists for JSON serialization
        state_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in state.items()}
        return jsonify(state_json)
    
    @app.route('/move_to_joint_positions', methods=['POST'])
    def move_to_joint_positions():
        data = request.json
        positions = np.array(data['positions'])
        success = franka_server.move_to_joint_positions(
            positions,
            velocity=data.get('velocity'),
            acceleration=data.get('acceleration'),
            blocking=data.get('blocking', True),
        )
        return jsonify({'success': success})
    
    @app.route('/send_cartesian_command', methods=['POST'])
    def send_cartesian_command():
        data = request.json
        position = np.array(data['position'])
        orientation = np.array(data['orientation']) if 'orientation' in data else None
        success = franka_server.send_cartesian_command(position, orientation)
        return jsonify({'success': success})
    
    @app.route('/reset', methods=['POST'])
    def reset():
        success = franka_server.reset()
        return jsonify({'success': success})
    
    @app.route('/stop', methods=['POST'])
    def stop():
        success = franka_server.stop()
        return jsonify({'success': success})
    
    @app.route('/error_recovery', methods=['POST'])
    def error_recovery():
        success = franka_server.error_recovery()
        return jsonify({'success': success})
    
    return app


def main(argv):
    # Create Franka server
    franka_server = FrankaServer(
        robot_ip=FLAGS.robot_ip,
        gripper_type=FLAGS.gripper_type,
        reset_joint_target=[float(x) for x in FLAGS.reset_joint_target],
    )
    
    # Connect to robot
    if franka_server.connect():
        print("[FrankaServer] Successfully connected to robot")
    else:
        print("[FrankaServer] Failed to connect to robot")
        return
    
    # Create and run Flask app
    flask_app = create_app(franka_server)
    print("[FrankaServer] Starting HTTP server on port 5000")
    flask_app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    app.run(main)
