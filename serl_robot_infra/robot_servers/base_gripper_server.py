"""
Base abstract interface for gripper servers.
This replaces ROS-based communication with a pure Python SDK interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseGripperServer(ABC):
    """
    Abstract base class for gripper control servers.
    
    This interface should be implemented for specific grippers (Robotiq, Franka Hand, etc.)
    using their respective Python SDKs instead of ROS.
    """
    
    @abstractmethod
    def __init__(self, gripper_ip: Optional[str] = None, **kwargs):
        """
        Initialize gripper connection.
        
        Args:
            gripper_ip: IP address of the gripper (if standalone)
            **kwargs: Additional gripper-specific configuration
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the gripper.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the gripper.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def open(self, speed: Optional[float] = None, force: Optional[float] = None, blocking: bool = True) -> bool:
        """
        Open the gripper.
        
        Args:
            speed: Opening speed (if None, use default)
            force: Maximum force (if None, use default)
            blocking: Whether to wait for motion to complete
            
        Returns:
            True if command successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self, speed: Optional[float] = None, force: Optional[float] = None, blocking: bool = True) -> bool:
        """
        Close the gripper.
        
        Args:
            speed: Closing speed (if None, use default)
            force: Maximum force (if None, use default)
            blocking: Whether to wait for motion to complete
            
        Returns:
            True if command successful, False otherwise
        """
        pass
    
    @abstractmethod
    def move_to_position(self, position: float, speed: Optional[float] = None, force: Optional[float] = None, blocking: bool = True) -> bool:
        """
        Move gripper to a specific position.
        
        Args:
            position: Target position (normalized 0-1 or in meters, depending on gripper)
            speed: Movement speed (if None, use default)
            force: Maximum force (if None, use default)
            blocking: Whether to wait for motion to complete
            
        Returns:
            True if command successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_state(self) -> dict:
        """
        Get current gripper state.
        
        Returns:
            Dictionary containing:
                - 'position': Current gripper position
                - 'is_moving': Whether gripper is currently moving
                - 'force': Current force (if available)
                - 'is_grasping': Whether gripper is grasping an object
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop gripper motion immediately.
        
        Returns:
            True if stop successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if gripper is connected and responsive.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """
        Reset/calibrate the gripper.
        
        Returns:
            True if reset successful, False otherwise
        """
        pass
    
    def get_limits(self) -> tuple:
        """
        Get gripper position limits (optional).
        
        Returns:
            Tuple of (min_position, max_position)
        """
        return (0.0, 1.0)
    
    def grasp(self, force: Optional[float] = None, blocking: bool = True) -> bool:
        """
        Execute a grasp action (convenience method).
        
        Args:
            force: Grasping force (if None, use default)
            blocking: Whether to wait for grasp to complete
            
        Returns:
            True if grasp successful, False otherwise
        """
        return self.close(force=force, blocking=blocking)
    
    def release(self, blocking: bool = True) -> bool:
        """
        Release a grasp (convenience method).
        
        Args:
            blocking: Whether to wait for release to complete
            
        Returns:
            True if release successful, False otherwise
        """
        return self.open(blocking=blocking)

