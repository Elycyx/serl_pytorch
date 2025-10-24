"""
Base abstract interface for robot servers.
This replaces ROS-based communication with a pure Python SDK interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BaseRobotServer(ABC):
    """
    Abstract base class for robot control servers.
    
    This interface should be implemented for specific robots (Franka, UR, etc.)
    using their respective Python SDKs instead of ROS.
    """
    
    @abstractmethod
    def __init__(self, robot_ip: str, **kwargs):
        """
        Initialize robot connection.
        
        Args:
            robot_ip: IP address of the robot
            **kwargs: Additional robot-specific configuration
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the robot.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the robot.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current robot state.
        
        Returns:
            Dictionary containing:
                - 'joint_positions': Current joint positions
                - 'joint_velocities': Current joint velocities
                - 'joint_torques': Current joint torques (if available)
                - 'cartesian_position': End-effector position [x, y, z]
                - 'cartesian_orientation': End-effector orientation [qx, qy, qz, qw]
        """
        pass
    
    @abstractmethod
    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move robot to target joint positions.
        
        Args:
            positions: Target joint positions
            velocity: Maximum velocity (if None, use default)
            acceleration: Maximum acceleration (if None, use default)
            blocking: Whether to wait for motion to complete
            
        Returns:
            True if command accepted, False otherwise
        """
        pass
    
    @abstractmethod
    def move_to_cartesian_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move robot end-effector to target Cartesian pose.
        
        Args:
            position: Target position [x, y, z]
            orientation: Target orientation [qx, qy, qz, qw] or rotation matrix
            velocity: Maximum velocity (if None, use default)
            acceleration: Maximum acceleration (if None, use default)
            blocking: Whether to wait for motion to complete
            
        Returns:
            True if command accepted, False otherwise
        """
        pass
    
    
    @abstractmethod
    def reset(self) -> bool:
        """
        Reset robot to home/safe position.
        
        Returns:
            True if reset successful, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Emergency stop - halt all robot motion immediately.
        
        Returns:
            True if stop successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if robot is connected and responsive.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get robot joint limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        pass
    
    @abstractmethod
    def get_workspace_limits(self) -> Dict[str, np.ndarray]:
        """
        Get robot workspace limits.
        
        Returns:
            Dictionary with 'position_min', 'position_max'
        """
        pass
    
    def get_jacobian(self) -> Optional[np.ndarray]:
        """
        Get current Jacobian matrix (optional).
        
        Returns:
            Jacobian matrix if available, None otherwise
        """
        return None
    
    def get_mass_matrix(self) -> Optional[np.ndarray]:
        """
        Get current mass matrix (optional).
        
        Returns:
            Mass matrix if available, None otherwise
        """
        return None
    
    def set_control_mode(self, mode: str) -> bool:
        """
        Set robot control mode (optional).
        
        Args:
            mode: Control mode ('position', 'velocity', 'torque', 'impedance', etc.)
            
        Returns:
            True if mode set successfully, False otherwise
        """
        return False

