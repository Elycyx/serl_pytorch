# 🤖 新机器人适配指南

**版本**: PyTorch SERL v1.0  
**更新日期**: 2025年10月20日

---

## 📚 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [详细步骤](#详细步骤)
4. [常见机器人SDK适配](#常见机器人sdk适配)
5. [完整示例](#完整示例)
6. [测试和调试](#测试和调试)
7. [常见问题](#常见问题)

---

## 概述

### 架构设计

SERL PyTorch 版本使用**抽象基类**设计，完全移除了 ROS 依赖。这使得适配新机器人变得简单且灵活。

```
serl_robot_infra/
├── robot_servers/
│   ├── base_robot_server.py      # 机器人抽象基类 ⭐
│   ├── base_gripper_server.py    # 夹爪抽象基类 ⭐
│   ├── franka_server.py           # Franka 实现示例 📖
│   ├── franka_gripper_server.py  # Franka 夹爪示例 📖
│   ├── robotiq_gripper_server.py # Robotiq 夹爪示例 📖
│   └── your_robot_server.py      # 你的机器人实现 ✍️
```

### 核心原则

1. ✅ **继承抽象基类** - 确保接口统一
2. ✅ **实现所有抽象方法** - 保证功能完整
3. ✅ **使用机器人SDK** - 直接调用厂商SDK，无需ROS
4. ✅ **遵循类型注解** - 确保数据类型正确

---

## 快速开始

### 第一步：安装机器人SDK

```bash
# 示例：UR机器人
pip install ur-rtde

# 示例：ABB机器人
pip install abb-robot-driver

# 示例：KUKA机器人
pip install kuka-sunrise-python
```

### 第二步：创建机器人服务器类

```python
# serl_robot_infra/robot_servers/your_robot_server.py

from serl_robot_infra.robot_servers.base_robot_server import BaseRobotServer
import numpy as np
from typing import Dict, Optional, Tuple

class YourRobotServer(BaseRobotServer):
    """你的机器人控制服务器"""
    
    def __init__(self, robot_ip: str, **kwargs):
        self.robot_ip = robot_ip
        # 初始化你的机器人SDK
        # self.robot = YourRobotSDK(robot_ip)
        
    def connect(self) -> bool:
        # 连接到机器人
        # return self.robot.connect()
        pass
    
    # ... 实现其他方法
```

### 第三步：实现夹爪服务器（如需要）

```python
# serl_robot_infra/robot_servers/your_gripper_server.py

from serl_robot_infra.robot_servers.base_gripper_server import BaseGripperServer

class YourGripperServer(BaseGripperServer):
    """你的夹爪控制服务器"""
    
    def __init__(self, gripper_ip: Optional[str] = None, **kwargs):
        # 初始化夹爪
        pass
    
    # ... 实现其他方法
```

---

## 详细步骤

### 步骤1：理解抽象基类

#### BaseRobotServer 必须实现的方法

| 方法 | 功能 | 返回值 | 重要性 |
|------|------|--------|--------|
| `connect()` | 连接机器人 | `bool` | ⭐⭐⭐ |
| `disconnect()` | 断开连接 | `bool` | ⭐⭐⭐ |
| `get_state()` | 获取状态 | `Dict` | ⭐⭐⭐ |
| `move_to_joint_positions()` | 关节运动 | `bool` | ⭐⭐⭐ |
| `move_to_cartesian_pose()` | 笛卡尔运动 | `bool` | ⭐⭐⭐ |
| `send_joint_command()` | 发送关节命令 | `bool` | ⭐⭐⭐ |
| `send_cartesian_command()` | 发送笛卡尔命令 | `bool` | ⭐⭐ |
| `reset()` | 复位 | `bool` | ⭐⭐⭐ |
| `stop()` | 紧急停止 | `bool` | ⭐⭐⭐ |
| `is_connected()` | 检查连接 | `bool` | ⭐⭐⭐ |
| `get_joint_limits()` | 获取关节限制 | `Tuple` | ⭐⭐ |
| `get_workspace_limits()` | 获取工作空间 | `Dict` | ⭐⭐ |

#### BaseGripperServer 必须实现的方法

| 方法 | 功能 | 返回值 | 重要性 |
|------|------|--------|--------|
| `connect()` | 连接夹爪 | `bool` | ⭐⭐⭐ |
| `disconnect()` | 断开连接 | `bool` | ⭐⭐⭐ |
| `open()` | 打开夹爪 | `bool` | ⭐⭐⭐ |
| `close()` | 关闭夹爪 | `bool` | ⭐⭐⭐ |
| `move_to_position()` | 移动到位置 | `bool` | ⭐⭐⭐ |
| `get_state()` | 获取夹爪状态 | `dict` | ⭐⭐⭐ |
| `stop()` | 停止 | `bool` | ⭐⭐⭐ |
| `is_connected()` | 检查连接 | `bool` | ⭐⭐⭐ |
| `reset()` | 复位 | `bool` | ⭐⭐ |

---

### 步骤2：实现核心功能

#### 2.1 连接管理

```python
class YourRobotServer(BaseRobotServer):
    def __init__(self, robot_ip: str, **kwargs):
        self.robot_ip = robot_ip
        self._is_connected = False
        
        # 初始化SDK
        try:
            # 示例：使用UR机器人
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
            
            self.control = RTDEControlInterface(robot_ip)
            self.receive = RTDEReceiveInterface(robot_ip)
            print(f"[YourRobot] Initialized SDK for {robot_ip}")
        except Exception as e:
            print(f"[YourRobot] SDK initialization failed: {e}")
            self.control = None
            self.receive = None
    
    def connect(self) -> bool:
        """建立连接"""
        try:
            if self.control is None or self.receive is None:
                return False
            
            # 检查连接
            self._is_connected = self.receive.isConnected()
            
            if self._is_connected:
                print("[YourRobot] Connected successfully")
            else:
                print("[YourRobot] Connection failed")
            
            return self._is_connected
        except Exception as e:
            print(f"[YourRobot] Connection error: {e}")
            self._is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self.control is not None:
                self.control.disconnect()
            if self.receive is not None:
                self.receive.disconnect()
            
            self._is_connected = False
            print("[YourRobot] Disconnected")
            return True
        except Exception as e:
            print(f"[YourRobot] Disconnect error: {e}")
            return False
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        if self.receive is not None:
            self._is_connected = self.receive.isConnected()
        return self._is_connected
```

#### 2.2 状态获取

```python
def get_state(self) -> Dict[str, np.ndarray]:
    """
    获取机器人状态
    
    必须返回以下键：
    - 'joint_positions': 关节位置
    - 'joint_velocities': 关节速度
    - 'joint_torques': 关节力矩（可选）
    - 'cartesian_position': 末端位置 [x, y, z]
    - 'cartesian_orientation': 末端姿态 [qx, qy, qz, qw]
    """
    try:
        # 获取关节状态
        joint_positions = np.array(self.receive.getActualQ())
        joint_velocities = np.array(self.receive.getActualQd())
        
        # 获取末端位姿
        tcp_pose = self.receive.getActualTCPPose()  # [x, y, z, rx, ry, rz]
        
        # 转换旋转向量到四元数
        from scipy.spatial.transform import Rotation as R
        rotation_vector = tcp_pose[3:]
        rotation = R.from_rotvec(rotation_vector)
        quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
        
        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'joint_torques': np.zeros(len(joint_positions)),  # 如果不可用
            'cartesian_position': np.array(tcp_pose[:3]),
            'cartesian_orientation': quaternion,
        }
    except Exception as e:
        print(f"[YourRobot] Get state error: {e}")
        return {
            'joint_positions': np.zeros(6),
            'joint_velocities': np.zeros(6),
            'joint_torques': np.zeros(6),
            'cartesian_position': np.zeros(3),
            'cartesian_orientation': np.array([0, 0, 0, 1]),
        }
```

#### 2.3 运动控制

```python
def move_to_joint_positions(
    self,
    positions: np.ndarray,
    velocity: Optional[float] = None,
    acceleration: Optional[float] = None,
    blocking: bool = True,
) -> bool:
    """移动到目标关节位置"""
    try:
        if not self.is_connected():
            print("[YourRobot] Not connected")
            return False
        
        # 设置默认速度和加速度
        if velocity is None:
            velocity = 0.5  # rad/s
        if acceleration is None:
            acceleration = 0.5  # rad/s^2
        
        # 转换为列表
        target_positions = positions.tolist()
        
        # 发送运动命令
        success = self.control.moveJ(
            target_positions,
            speed=velocity,
            acceleration=acceleration,
            asynchronous=not blocking
        )
        
        if blocking:
            # 等待运动完成
            while not self._is_at_target(target_positions):
                time.sleep(0.01)
        
        return success
    except Exception as e:
        print(f"[YourRobot] Move error: {e}")
        return False

def move_to_cartesian_pose(
    self,
    position: np.ndarray,
    orientation: np.ndarray,
    velocity: Optional[float] = None,
    acceleration: Optional[float] = None,
    blocking: bool = True,
) -> bool:
    """移动末端到目标笛卡尔位姿"""
    try:
        if not self.is_connected():
            return False
        
        # 设置默认参数
        if velocity is None:
            velocity = 0.25  # m/s
        if acceleration is None:
            acceleration = 1.2  # m/s^2
        
        # 转换四元数到旋转向量
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(orientation)
        rotation_vector = rotation.as_rotvec()
        
        # 组合位姿 [x, y, z, rx, ry, rz]
        target_pose = np.concatenate([position, rotation_vector]).tolist()
        
        # 发送运动命令
        success = self.control.moveL(
            target_pose,
            speed=velocity,
            acceleration=acceleration,
            asynchronous=not blocking
        )
        
        return success
    except Exception as e:
        print(f"[YourRobot] Move cartesian error: {e}")
        return False

def send_joint_command(
    self,
    command: np.ndarray,
    command_type: str = 'position',
) -> bool:
    """发送低级关节命令（用于RL控制）"""
    try:
        if not self.is_connected():
            return False
        
        command_list = command.tolist()
        
        if command_type == 'position':
            # 位置控制
            return self.control.servoJ(command_list, lookahead_time=0.1, gain=300)
        elif command_type == 'velocity':
            # 速度控制
            return self.control.speedJ(command_list, acceleration=1.4)
        elif command_type == 'torque':
            # 力矩控制（如果支持）
            print("[YourRobot] Torque control not implemented")
            return False
        else:
            print(f"[YourRobot] Unknown command type: {command_type}")
            return False
    except Exception as e:
        print(f"[YourRobot] Send command error: {e}")
        return False
```

#### 2.4 安全功能

```python
def reset(self) -> bool:
    """复位到安全位置"""
    try:
        # 定义安全/home位置
        home_position = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
        
        print("[YourRobot] Resetting to home position...")
        success = self.move_to_joint_positions(
            home_position,
            velocity=0.5,
            acceleration=0.5,
            blocking=True
        )
        
        if success:
            print("[YourRobot] Reset complete")
        else:
            print("[YourRobot] Reset failed")
        
        return success
    except Exception as e:
        print(f"[YourRobot] Reset error: {e}")
        return False

def stop(self) -> bool:
    """紧急停止"""
    try:
        if self.control is not None:
            self.control.stopScript()
            print("[YourRobot] Emergency stop executed")
            return True
        return False
    except Exception as e:
        print(f"[YourRobot] Stop error: {e}")
        return False
```

#### 2.5 参数获取

```python
def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
    """获取关节限制"""
    # 根据你的机器人规格定义
    # 示例：UR5e
    lower_limits = np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
    upper_limits = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
    
    return lower_limits, upper_limits

def get_workspace_limits(self) -> Dict[str, np.ndarray]:
    """获取工作空间限制"""
    # 根据你的机器人规格定义
    # 示例：UR5e (单位：米)
    return {
        'position_min': np.array([-0.8, -0.8, 0.0]),
        'position_max': np.array([0.8, 0.8, 1.0]),
    }

def get_jacobian(self) -> Optional[np.ndarray]:
    """获取雅可比矩阵（可选）"""
    try:
        # 如果SDK支持
        # jacobian = self.receive.getJacobian()
        # return np.array(jacobian)
        return None
    except:
        return None
```

---

### 步骤3：实现夹爪服务器

```python
from serl_robot_infra.robot_servers.base_gripper_server import BaseGripperServer

class YourGripperServer(BaseGripperServer):
    """你的夹爪服务器实现"""
    
    def __init__(self, gripper_ip: Optional[str] = None, **kwargs):
        self.gripper_ip = gripper_ip
        self._is_connected = False
        
        # 初始化夹爪SDK
        # 示例：Robotiq 2F-85
        # from robotiq_2f_gripper_control import Robotiq2FGripper
        # self.gripper = Robotiq2FGripper(gripper_ip)
    
    def connect(self) -> bool:
        """连接夹爪"""
        try:
            # self._is_connected = self.gripper.connect()
            self._is_connected = True  # Placeholder
            return self._is_connected
        except Exception as e:
            print(f"[Gripper] Connect error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开连接"""
        try:
            # self.gripper.disconnect()
            self._is_connected = False
            return True
        except Exception as e:
            print(f"[Gripper] Disconnect error: {e}")
            return False
    
    def open(self, speed: Optional[float] = None, force: Optional[float] = None, blocking: bool = True) -> bool:
        """打开夹爪"""
        try:
            if not self.is_connected():
                return False
            
            # 使用默认值
            if speed is None:
                speed = 0.1  # 0-1
            if force is None:
                force = 0.5  # 0-1
            
            # 发送打开命令
            # success = self.gripper.move(position=1.0, speed=speed, force=force)
            success = True  # Placeholder
            
            if blocking:
                # 等待完成
                # while self.gripper.is_moving():
                #     time.sleep(0.01)
                pass
            
            return success
        except Exception as e:
            print(f"[Gripper] Open error: {e}")
            return False
    
    def close(self, speed: Optional[float] = None, force: Optional[float] = None, blocking: bool = True) -> bool:
        """关闭夹爪"""
        try:
            if not self.is_connected():
                return False
            
            if speed is None:
                speed = 0.1
            if force is None:
                force = 0.5
            
            # 发送关闭命令
            # success = self.gripper.move(position=0.0, speed=speed, force=force)
            success = True
            
            if blocking:
                # 等待完成
                pass
            
            return success
        except Exception as e:
            print(f"[Gripper] Close error: {e}")
            return False
    
    def move_to_position(self, position: float, speed: Optional[float] = None, force: Optional[float] = None, blocking: bool = True) -> bool:
        """移动到指定位置"""
        try:
            if not self.is_connected():
                return False
            
            # 限制范围 [0, 1]
            position = np.clip(position, 0.0, 1.0)
            
            if speed is None:
                speed = 0.1
            if force is None:
                force = 0.5
            
            # success = self.gripper.move(position=position, speed=speed, force=force)
            success = True
            
            return success
        except Exception as e:
            print(f"[Gripper] Move error: {e}")
            return False
    
    def get_state(self) -> dict:
        """获取夹爪状态"""
        try:
            # state = self.gripper.get_state()
            return {
                'position': 0.5,  # 当前位置 [0, 1]
                'is_moving': False,  # 是否运动中
                'force': 0.0,  # 当前力
                'is_grasping': False,  # 是否抓取物体
            }
        except Exception as e:
            print(f"[Gripper] Get state error: {e}")
            return {
                'position': 0.0,
                'is_moving': False,
                'force': 0.0,
                'is_grasping': False,
            }
    
    def stop(self) -> bool:
        """停止夹爪"""
        try:
            # self.gripper.stop()
            return True
        except Exception as e:
            print(f"[Gripper] Stop error: {e}")
            return False
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._is_connected
    
    def reset(self) -> bool:
        """复位夹爪"""
        try:
            # 通常是重新激活/校准
            # self.gripper.activate()
            return True
        except Exception as e:
            print(f"[Gripper] Reset error: {e}")
            return False
```

---

## 常见机器人SDK适配

### 1. Universal Robots (UR)

**SDK**: `ur-rtde`

```bash
pip install ur-rtde
```

**示例代码**:
```python
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

class URRobotServer(BaseRobotServer):
    def __init__(self, robot_ip: str, **kwargs):
        self.control = RTDEControlInterface(robot_ip)
        self.receive = RTDEReceiveInterface(robot_ip)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        joint_positions = np.array(self.receive.getActualQ())
        joint_velocities = np.array(self.receive.getActualQd())
        tcp_pose = self.receive.getActualTCPPose()
        
        # ... 转换格式
        return state_dict
```

**文档**: https://sdurobotics.gitlab.io/ur_rtde/

---

### 2. ABB Robots

**SDK**: `abb-robot-driver` 或 `compas_rrc`

```bash
pip install compas_rrc
```

**示例代码**:
```python
from compas_rrc import AbbClient, MoveToJoints

class ABBRobotServer(BaseRobotServer):
    def __init__(self, robot_ip: str, **kwargs):
        self.client = AbbClient(robot_ip)
        self.client.run()
    
    def move_to_joint_positions(self, positions: np.ndarray, **kwargs) -> bool:
        move_cmd = MoveToJoints(positions.tolist())
        self.client.send(move_cmd)
        return True
```

**文档**: https://compas.dev/compas_rrc/

---

### 3. KUKA Robots

**SDK**: `kuka_sunrise` 或 `pybotics`

```bash
pip install kuka-sunrise
```

**示例代码**:
```python
from kuka_sunrise import KUKAiiwaInterface

class KUKARobotServer(BaseRobotServer):
    def __init__(self, robot_ip: str, **kwargs):
        self.robot = KUKAiiwaInterface(robot_ip)
        self.robot.start()
    
    def send_joint_command(self, command: np.ndarray, **kwargs) -> bool:
        self.robot.send_joint_position(command.tolist())
        return True
```

---

### 4. Franka Emika (Panda)

**SDK**: `libfranka-python`

```bash
# 需要从源码编译
git clone https://github.com/frankaemika/libfranka
cd libfranka
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON
make
sudo make install
```

**参考**: 已有 `franka_server.py` 模板实现

---

### 5. Kinova Robots

**SDK**: `kortex_api`

```bash
pip install kortex-api
```

**示例代码**:
```python
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

class KinovaRobotServer(BaseRobotServer):
    def __init__(self, robot_ip: str, **kwargs):
        # 初始化Kortex API
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.SessionManager import SessionManager
        
        self.transport = TCPTransport()
        self.router = self.transport.connect(robot_ip, 10000)
        self.session = SessionManager(self.router)
        self.base = BaseClient(self.router)
```

**文档**: https://github.com/Kinovahttps://github.com/Kinova/kortex

---

### 6. 自定义/其他机器人

如果你的机器人有TCP/IP接口但没有Python SDK：

```python
import socket
import struct

class CustomRobotServer(BaseRobotServer):
    def __init__(self, robot_ip: str, port: int = 30003, **kwargs):
        self.robot_ip = robot_ip
        self.port = port
        self.socket = None
    
    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.robot_ip, self.port))
            self.socket.settimeout(1.0)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def send_joint_command(self, command: np.ndarray, **kwargs) -> bool:
        try:
            # 按照你的协议打包数据
            data = struct.pack('6d', *command)
            self.socket.send(data)
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def get_state(self) -> Dict[str, np.ndarray]:
        try:
            # 接收数据
            data = self.socket.recv(1024)
            # 解析数据
            values = struct.unpack('6d', data[:48])
            joint_positions = np.array(values)
            
            return {
                'joint_positions': joint_positions,
                # ... 其他状态
            }
        except Exception as e:
            print(f"Receive error: {e}")
            return {}
```

---

## 完整示例

### 完整的UR机器人适配

```python
# serl_robot_infra/robot_servers/ur_robot_server.py

"""
Universal Robots (UR3/UR5/UR10/UR16) server implementation.
Uses ur-rtde for direct robot control without ROS.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

from serl_robot_infra.robot_servers.base_robot_server import BaseRobotServer

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    UR_RTDE_AVAILABLE = True
except ImportError:
    UR_RTDE_AVAILABLE = False
    print("Warning: ur-rtde not installed. Install with: pip install ur-rtde")


class URRobotServer(BaseRobotServer):
    """
    Universal Robots control server using ur-rtde.
    
    Supports: UR3, UR3e, UR5, UR5e, UR10, UR10e, UR16e
    """
    
    def __init__(
        self,
        robot_ip: str,
        home_joint_target: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initialize UR robot server.
        
        Args:
            robot_ip: IP address of UR robot
            home_joint_target: Home position joint angles
            **kwargs: Additional configuration
        """
        if not UR_RTDE_AVAILABLE:
            raise RuntimeError("ur-rtde is not installed")
        
        self.robot_ip = robot_ip
        self.home_joint_target = home_joint_target or np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
        
        # Initialize RTDE interfaces
        try:
            self.control = RTDEControlInterface(robot_ip)
            self.receive = RTDEReceiveInterface(robot_ip)
            self._is_connected = True
            print(f"[URRobot] Connected to {robot_ip}")
        except Exception as e:
            print(f"[URRobot] Initialization failed: {e}")
            self.control = None
            self.receive = None
            self._is_connected = False
    
    def connect(self) -> bool:
        """Establish connection to UR robot."""
        if self.control is None or self.receive is None:
            try:
                self.control = RTDEControlInterface(self.robot_ip)
                self.receive = RTDEReceiveInterface(self.robot_ip)
                self._is_connected = True
                print(f"[URRobot] Reconnected to {self.robot_ip}")
            except Exception as e:
                print(f"[URRobot] Connection failed: {e}")
                self._is_connected = False
        
        return self._is_connected
    
    def disconnect(self) -> bool:
        """Disconnect from UR robot."""
        try:
            if self.control:
                self.control.stopScript()
            
            # RTDE interfaces disconnect automatically
            self.control = None
            self.receive = None
            self._is_connected = False
            print("[URRobot] Disconnected")
            return True
        except Exception as e:
            print(f"[URRobot] Disconnect error: {e}")
            return False
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current robot state."""
        try:
            # Joint state
            joint_positions = np.array(self.receive.getActualQ())
            joint_velocities = np.array(self.receive.getActualQd())
            
            # TCP pose [x, y, z, rx, ry, rz]
            tcp_pose = self.receive.getActualTCPPose()
            
            # Convert rotation vector to quaternion
            rotation_vector = np.array(tcp_pose[3:])
            rotation = R.from_rotvec(rotation_vector)
            quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
            
            return {
                'joint_positions': joint_positions,
                'joint_velocities': joint_velocities,
                'joint_torques': np.zeros(6),  # UR doesn't provide direct torque readout
                'cartesian_position': np.array(tcp_pose[:3]),
                'cartesian_orientation': quaternion,
            }
        except Exception as e:
            print(f"[URRobot] Get state error: {e}")
            return {
                'joint_positions': np.zeros(6),
                'joint_velocities': np.zeros(6),
                'joint_torques': np.zeros(6),
                'cartesian_position': np.zeros(3),
                'cartesian_orientation': np.array([0, 0, 0, 1]),
            }
    
    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """Move to target joint positions."""
        try:
            if not self.is_connected():
                return False
            
            velocity = velocity or 0.5  # rad/s
            acceleration = acceleration or 0.5  # rad/s^2
            
            target = positions.tolist()
            
            success = self.control.moveJ(
                target,
                speed=velocity,
                acceleration=acceleration,
                asynchronous=not blocking
            )
            
            return success
        except Exception as e:
            print(f"[URRobot] Move joint error: {e}")
            return False
    
    def move_to_cartesian_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """Move end-effector to target Cartesian pose."""
        try:
            if not self.is_connected():
                return False
            
            velocity = velocity or 0.25  # m/s
            acceleration = acceleration or 1.2  # m/s^2
            
            # Convert quaternion to rotation vector
            rotation = R.from_quat(orientation)
            rotation_vector = rotation.as_rotvec()
            
            # Combine to pose [x, y, z, rx, ry, rz]
            target_pose = np.concatenate([position, rotation_vector]).tolist()
            
            success = self.control.moveL(
                target_pose,
                speed=velocity,
                acceleration=acceleration,
                asynchronous=not blocking
            )
            
            return success
        except Exception as e:
            print(f"[URRobot] Move cartesian error: {e}")
            return False
    
    def send_joint_command(
        self,
        command: np.ndarray,
        command_type: str = 'position',
    ) -> bool:
        """Send low-level joint command (for RL control)."""
        try:
            if not self.is_connected():
                return False
            
            cmd = command.tolist()
            
            if command_type == 'position':
                # Servo to position
                return self.control.servoJ(cmd, lookahead_time=0.1, gain=300)
            elif command_type == 'velocity':
                # Velocity control
                return self.control.speedJ(cmd, acceleration=1.4)
            else:
                print(f"[URRobot] Unsupported command type: {command_type}")
                return False
        except Exception as e:
            print(f"[URRobot] Send command error: {e}")
            return False
    
    def send_cartesian_command(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> bool:
        """Send Cartesian space command."""
        try:
            if not self.is_connected():
                return False
            
            if orientation is None:
                # Keep current orientation
                current_pose = self.receive.getActualTCPPose()
                orientation_vec = current_pose[3:]
            else:
                # Convert quaternion to rotation vector
                rotation = R.from_quat(orientation)
                orientation_vec = rotation.as_rotvec()
            
            target_pose = np.concatenate([position, orientation_vec]).tolist()
            
            return self.control.servoL(target_pose, lookahead_time=0.1, gain=300)
        except Exception as e:
            print(f"[URRobot] Send cartesian command error: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset to home position."""
        print("[URRobot] Resetting to home position...")
        return self.move_to_joint_positions(
            self.home_joint_target,
            velocity=0.5,
            acceleration=0.5,
            blocking=True
        )
    
    def stop(self) -> bool:
        """Emergency stop."""
        try:
            if self.control:
                self.control.stopScript()
                print("[URRobot] Emergency stop executed")
                return True
            return False
        except Exception as e:
            print(f"[URRobot] Stop error: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check connection status."""
        try:
            if self.receive:
                return self.receive.isConnected()
            return False
        except:
            return False
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits."""
        # UR robot joint limits
        lower = np.array([-2*np.pi] * 6)
        upper = np.array([2*np.pi] * 6)
        return lower, upper
    
    def get_workspace_limits(self) -> Dict[str, np.ndarray]:
        """Get workspace limits (approximate for UR5)."""
        return {
            'position_min': np.array([-0.85, -0.85, 0.0]),
            'position_max': np.array([0.85, 0.85, 1.0]),
        }
    
    def get_jacobian(self) -> Optional[np.ndarray]:
        """Get current Jacobian matrix."""
        try:
            jacobian = self.receive.getActualJacobian()
            return np.array(jacobian)
        except:
            return None


# 使用示例
if __name__ == "__main__":
    robot = URRobotServer("192.168.1.100")
    
    if robot.connect():
        print("Connected!")
        
        # 获取状态
        state = robot.get_state()
        print(f"Joint positions: {state['joint_positions']}")
        
        # 复位
        robot.reset()
        
        # 断开
        robot.disconnect()
```

---

## 测试和调试

### 1. 单元测试

创建测试文件 `test_your_robot.py`:

```python
import unittest
import numpy as np
from serl_robot_infra.robot_servers.your_robot_server import YourRobotServer

class TestYourRobot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = YourRobotServer("192.168.1.100")
        cls.robot.connect()
    
    def test_connection(self):
        self.assertTrue(self.robot.is_connected())
    
    def test_get_state(self):
        state = self.robot.get_state()
        self.assertIn('joint_positions', state)
        self.assertEqual(len(state['joint_positions']), 6)
    
    def test_move_joint(self):
        target = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
        success = self.robot.move_to_joint_positions(target)
        self.assertTrue(success)
    
    @classmethod
    def tearDownClass(cls):
        cls.robot.disconnect()

if __name__ == '__main__':
    unittest.main()
```

### 2. 调试技巧

#### 启用详细日志

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class YourRobotServer(BaseRobotServer):
    def send_joint_command(self, command, command_type='position'):
        logger.debug(f"Sending {command_type} command: {command}")
        # ...
```

#### 监控状态

```python
# 实时监控机器人状态
import time

robot = YourRobotServer("192.168.1.100")
robot.connect()

while True:
    state = robot.get_state()
    print(f"Joints: {state['joint_positions']}")
    print(f"TCP: {state['cartesian_position']}")
    time.sleep(0.1)
```

#### 安全模式测试

```python
# 限制运动范围测试
class SafeYourRobotServer(YourRobotServer):
    def send_joint_command(self, command, command_type='position'):
        # 检查限制
        lower, upper = self.get_joint_limits()
        if np.any(command < lower) or np.any(command > upper):
            print(f"Command out of limits: {command}")
            return False
        
        # 限制速度
        current_state = self.get_state()
        current_pos = current_state['joint_positions']
        delta = np.abs(command - current_pos)
        max_delta = 0.1  # rad
        
        if np.any(delta > max_delta):
            print(f"Command change too large: {delta}")
            return False
        
        return super().send_joint_command(command, command_type)
```

---

## 常见问题

### Q1: 如何处理机器人SDK没有某些功能？

**A**: 提供合理的默认实现或返回 None/False：

```python
def get_jacobian(self) -> Optional[np.ndarray]:
    """如果SDK不支持雅可比，返回None"""
    return None

def send_cartesian_command(self, position, orientation=None) -> bool:
    """如果不支持笛卡尔控制，使用逆运动学"""
    # 使用数值IK或返回False
    print("[Warning] Cartesian control not supported")
    return False
```

### Q2: 如何处理不同的旋转表示？

**A**: 使用 `scipy.spatial.transform.Rotation` 统一转换：

```python
from scipy.spatial.transform import Rotation as R

# 欧拉角 -> 四元数
euler = np.array([0, np.pi/2, 0])  # [roll, pitch, yaw]
rotation = R.from_euler('xyz', euler)
quaternion = rotation.as_quat()  # [qx, qy, qz, qw]

# 旋转向量 -> 四元数
rotvec = np.array([0, 1.57, 0])
rotation = R.from_rotvec(rotvec)
quaternion = rotation.as_quat()

# 旋转矩阵 -> 四元数
rotation_matrix = np.eye(3)
rotation = R.from_matrix(rotation_matrix)
quaternion = rotation.as_quat()
```

### Q3: 如何处理实时控制频率？

**A**: 在 `send_joint_command` 中使用高频控制：

```python
def send_joint_command(self, command, command_type='position'):
    # UR机器人：使用servoJ，125Hz
    if command_type == 'position':
        return self.control.servoJ(
            command.tolist(),
            lookahead_time=0.1,  # 100ms前瞻
            gain=300  # 控制增益
        )
```

### Q4: 如何集成到SERL环境？

**A**: 在环境中使用你的服务器：

```python
# serl_robot_infra/franka_env/envs/your_robot_env.py

from serl_robot_infra.robot_servers.your_robot_server import YourRobotServer
from serl_robot_infra.robot_servers.your_gripper_server import YourGripperServer

class YourRobotEnv(gym.Env):
    def __init__(self, robot_ip: str, gripper_ip: str = None, **kwargs):
        # 初始化机器人和夹爪
        self.robot = YourRobotServer(robot_ip)
        self.gripper = YourGripperServer(gripper_ip) if gripper_ip else None
        
        # 连接
        self.robot.connect()
        if self.gripper:
            self.gripper.connect()
    
    def step(self, action):
        # 发送动作
        self.robot.send_joint_command(action[:6])
        if self.gripper and len(action) > 6:
            self.gripper.move_to_position(action[6])
        
        # 获取观察
        state = self.robot.get_state()
        obs = state['joint_positions']
        
        # 计算奖励
        reward = self._compute_reward(obs, action)
        
        return obs, reward, False, False, {}
    
    def reset(self):
        self.robot.reset()
        if self.gripper:
            self.gripper.open()
        
        state = self.robot.get_state()
        return state['joint_positions'], {}
```

### Q5: 如何处理网络延迟？

**A**: 实现异步通信或预测控制：

```python
import threading
import queue

class AsyncRobotServer(BaseRobotServer):
    def __init__(self, robot_ip: str, **kwargs):
        super().__init__(robot_ip, **kwargs)
        self.command_queue = queue.Queue(maxsize=10)
        self.state_buffer = {}
        
        # 启动后台线程
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()
    
    def _control_loop(self):
        """后台控制循环"""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.01)
                # 实际发送命令
                self._send_command_internal(command)
            except queue.Empty:
                pass
            
            # 更新状态缓冲
            self.state_buffer = self._get_state_internal()
    
    def send_joint_command(self, command, command_type='position'):
        """非阻塞命令发送"""
        try:
            self.command_queue.put_nowait((command, command_type))
            return True
        except queue.Full:
            return False
    
    def get_state(self):
        """从缓冲返回最新状态"""
        return self.state_buffer
```

---

## 总结

### 核心步骤回顾

1. ✅ **安装机器人SDK**
2. ✅ **继承抽象基类** (`BaseRobotServer`, `BaseGripperServer`)
3. ✅ **实现所有抽象方法**
4. ✅ **测试基本功能** (连接、移动、获取状态)
5. ✅ **集成到环境** (在gym环境中使用)
6. ✅ **RL训练** (使用SAC等算法)

### 参考资源

- **抽象基类**: `serl_robot_infra/robot_servers/base_robot_server.py`
- **Franka示例**: `serl_robot_infra/robot_servers/franka_server.py`
- **UR示例**: 本文档中的完整示例
- **SERL论文**: https://serl-robot.github.io/

### 需要帮助？

如果遇到问题：
1. 检查机器人SDK文档
2. 参考 `franka_server.py` 实现
3. 查看单元测试示例
4. 使用调试模式运行

---

**祝你成功适配新机器人！🎉**

