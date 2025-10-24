"""

"""

from typing import Dict, Optional, Tuple
import numpy as np
import time
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation as R
from absl import app, flags
from piper_sdk import *

from serl_robot_infra.robot_servers.base_robot_server import BaseRobotServer



class PiperServer(BaseRobotServer):
    """
    Piper robot control server using direct SDK communication.
    """

    def __init__(self, can_name: str = 'can0', reset_joint_target: list = None, **kwargs):
        """
        Initialize Piper robot server.
        
        Args:
            can_name: Name of the CAN bus
            reset_joint_target: Target joint angles for reset position
            **kwargs: Additional configuration
        """
        self.can_name = can_name
        self.reset_joint_target = reset_joint_target or [0, 0, 0, 0, 0, 0, -1]
        
        # Robot state
        self._current_joint_positions = None
        self._current_cartesian_pose = None
        self._gripper_state = -1 # -1: close; 1: open
        self._is_connected = False
        
        self.robot = C_PiperInterface_V2(can_name)
        
        print(f"[PiperServer] Initialized for robot at {can_name}")
    
    def connect(self) -> bool:
        """
        Connect to the Piper robot.
        
        Returns:
            True if connection successful
        """
        try:
            self.robot.Connect()
            time.sleep(0.1)
            while(not self.robot.EnablePiper()):
                time.sleep(0.01)
            print("[PiperServer] Enabled Piper robot")
            self._is_connected = True
            return True
        except Exception as e:
            print(f"[PiperServer] Connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Piper robot.
        
        Returns:
            True if disconnection successful
        """
        try:
            print("[PiperServer] The robot will be disabled in 8 seconds")
            time.sleep(8)
            while(self.robot.DisablePiper()):
                time.sleep(0.01)
            print("[PiperServer] Disabled Piper robot")
            self._is_connected = False
            return True
        except Exception as e:
            print(f"[PiperServer] Disconnection failed: {e}")
            return False
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current robot state.
        
        Returns:
            Dictionary with joint positions, velocities, cartesian pose, etc.
        """
        joint_msgs = self.robot.GetArmJointMsgs()
        joint_state = joint_msgs.joint_state # in 0.001 degrees
        # transform to radians
        self._current_joint_positions = np.array([joint_state.joint_1 * np.pi / 180000, joint_state.joint_2 * np.pi / 180000, joint_state.joint_3 * np.pi / 180000, joint_state.joint_4 * np.pi / 180000, joint_state.joint_5 * np.pi / 180000, joint_state.joint_6 * np.pi / 180000])

        endpose_msgs = self.robot.GetArmEndPoseMsgs()
        endpose = endpose_msgs.end_pose # in 0.001 mm and 0.001 degrees
        # transform to meters and radians
        self._current_cartesian_pose = np.array([endpose.X_axis * 1e-6, endpose.Y_axis * 1e-6, endpose.Z_axis * 1e-6, endpose.RX_axis * np.pi / 180000, endpose.RY_axis * np.pi / 180000, endpose.RZ_axis * np.pi / 180000])

        gripper_msgs = self.robot.GetArmGripperMsgs() # in 0.001 mm
        gripper_state = gripper_msgs.gripper_state.grippers_angle # 0-80000
        if gripper_state <= 70000:
            self._gripper_state = -1 # close
        else:
            self._gripper_state = 1 # open

        return {
            'joint_positions': self._current_joint_positions,    
            'cartesian_pose': self._current_cartesian_pose,
            'gripper': self._gripper_state,
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
            positions: Target joint positions (6 values for Piper) in radians
            velocity: Maximum velocity
            acceleration: Maximum acceleration
            blocking: Whether to wait for completion
            
        Returns:
            True if successful
        """
        try:
            self.robot.MotionCtrl_2(0x01, 0x01, 100, 0x00) # switch to joint ctrl mode
            # from radians to 0.001 degrees
            joint_1 = int(positions[0] * 180000 / np.pi)
            joint_2 = int(positions[1] * 180000 / np.pi)
            joint_3 = int(positions[2] * 180000 / np.pi)
            joint_4 = int(positions[3] * 180000 / np.pi)
            joint_5 = int(positions[4] * 180000 / np.pi)
            joint_6 = int(positions[5] * 180000 / np.pi)
            self.robot.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
            if positions[-1] < 0:
                self.robot.GripperCtrl(0, 1000, 0x01, 0)
            else:
                self.robot.GripperCtrl(80000, 1000, 0x01, 0)
            
            print(f"[PiperServer] Moving to joint positions: {positions}")
            self._current_joint_positions = positions
            return True
        except Exception as e:
            print(f"[PiperServer] Joint motion failed: {e}")
            return False
    
    def move_to_cartesian_pose(
        self,
        pose: np.ndarray,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move to target Cartesian pose.
        
        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in meters and radians
            velocity: Maximum velocity
            acceleration: Maximum acceleration
            blocking: Whether to wait for completion
            
        Returns:
            True if successful
        """
        try:
            self.robot.MotionCtrl_2(0x01, 0x00, 100, 0x00) # switch to cartesian ctrl mode
            # from meters and radians to 0.001 mm and 0.001 degrees
            x = int(pose[0] * 1e6)
            y = int(pose[1] * 1e6)
            z = int(pose[2] * 1e6)
            rx = int(pose[3] * 180000 / np.pi)
            ry = int(pose[4] * 180000 / np.pi)
            rz = int(pose[5] * 180000 / np.pi)

            self.robot.EndPoseCtrl(x, y, z, rx, ry, rz)
            if positions[-1] < 0:
                self.robot.GripperCtrl(0, 1000, 0x01, 0)
            else:
                self.robot.GripperCtrl(80000, 1000, 0x01, 0)
            
            print(f"[PiperServer] Moving to Cartesian pose: pos={pose}")
            return True
        except Exception as e:
            print(f"[PiperServer] Cartesian motion failed: {e}")
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
            print(f"[PiperServer] Reset failed: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Emergency stop.
        
        Returns:
            True if successful
        """
        try:
            self.robot.disconnect()
            
            print("[PiperServer] Emergency stop called")
            return True
        except Exception as e:
            print(f"[PiperServer] Stop failed: {e}")
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
        Get Piper joint limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        # Piper joint limits (radians)
        lower = np.array([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439])
        upper = np.array([2.6179, 3.14, 0, 1.745, 1.22, 2.09439])
        return lower, upper
    
    def get_workspace_limits(self) -> Dict[str, np.ndarray]:
        """
        Get approximate workspace limits for Piper.
        
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
            6x6 Jacobian matrix
        """
        return None

def create_app(piper_server: PiperServer):
    """
    Create Flask app for HTTP API to robot server.
    
    Args:
        piper_server: Piper server instance
        
    Returns:
        Flask app
    """
    app = Flask(__name__)
    
    @app.route('/get_state', methods=['GET'])
    def get_state():
        state = piper_server.get_state()
        # Convert numpy arrays to lists for JSON serialization
        state_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in state.items()}
        return jsonify(state_json)
    
    @app.route('/move_to_joint_positions', methods=['POST'])
    def move_to_joint_positions():
        data = request.json
        positions = np.array(data['positions'])
        success = piper_server.move_to_joint_positions(
            positions,
            velocity=data.get('velocity'),
            acceleration=data.get('acceleration'),
            blocking=data.get('blocking', True),
        )
        return jsonify({'success': success})
    
    @app.route('/move_to_cartesian_pose', methods=['POST'])
    def move_to_cartesian_pose():
        data = request.json
        pose = np.array(data['pose'])
        success = piper_server.move_to_cartesian_pose(pose)
        return jsonify({'success': success})
    
    @app.route('/reset', methods=['POST'])
    def reset():
        success = piper_server.reset()
        return jsonify({'success': success})
    
    @app.route('/stop', methods=['POST'])
    def stop():
        success = piper_server.stop()
        return jsonify({'success': success})
    
    return app


def main(argv):
    # Create Piper server
    piper_server = PiperServer(
        can_name=FLAGS.can_name,
        reset_joint_target=[float(x) for x in FLAGS.reset_joint_target],
    )
    
    # Connect to robot
    if piper_server.connect():
        print("[PiperServer] Successfully connected to robot")
    else:
        print("[PiperServer] Failed to connect to robot")
        return
    
    # Create and run Flask app
    flask_app = create_app(piper_server)
    print("[PiperServer] Starting HTTP server on port 5000")
    flask_app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    app.run(main)
