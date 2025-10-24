<!-- dad9e06e-2476-45f7-98c8-5c006d364c19 164cacde-3747-4095-a343-da7e495aae63 -->
# SERL JAX 到 PyTorch 迁移计划

## 1. 核心神经网络模块转换 (serl_launcher/networks/)

将所有 Flax 网络定义转换为 PyTorch:

- **mlp.py**: MLP 网络从 `flax.linen.Module` 转换为 `torch.nn.Module`
- **actor_critic_nets.py**: Actor-Critic 网络架构
- **reward_classifier.py**: 奖励分类器
- **lagrange.py**: Lagrange 乘子网络
- **classifier.py**: 通用分类器

关键点：

- 使用 `torch.nn.Module` 替代 `flax.linen.Module`
- 参数初始化从 JAX 风格转换为 PyTorch `nn.init`
- 应用 `torch.compile()` 以获得类似 JAX JIT 的性能

## 2. RL 算法实现转换 (serl_launcher/agents/continuous/)

只转换 SAC 算法：

- **sac.py**: Soft Actor-Critic 算法

转换要点：

- JAX 的 `jax.jit` → PyTorch 的 `torch.compile()`
- JAX 的 `jax.vmap` → PyTorch 的 `torch.vmap()` 或批处理操作
- `optax` 优化器 → `torch.optim.AdamW`
- JAX 的 pytree 操作 → PyTorch 的嵌套张量或字典操作
- `flax.struct.dataclass` → Python `@dataclass` 与 PyTorch 状态字典
- 随机数生成：`jax.random.PRNGKey` → `torch.Generator`
- 保留 SAC 的所有核心功能：actor、critic、temperature 参数

注意：其他算法（drq.py、bc.py、vice.py）暂不转换，保留原文件但不修改

## 3. 视觉编码器转换 (serl_launcher/vision/)

转换所有视觉相关模块：

- **resnet_v1.py**: ResNet 编码器
- **mobilenet.py**: MobileNet 编码器
- **small_encoders.py**: 小型编码器
- **data_augmentations.py**: 数据增强（DrQ 风格）
- **film_conditioning_layer.py**: FiLM 条件层
- **spatial.py**: 空间 softmax

关键优化：

- 使用 `torch.channels_last` 内存格式提升卷积性能
- 应用 `torch.cuda.amp` 混合精度训练
- 使用 `torch.compile()` 编译视觉网络

## 4. 数据处理模块转换 (serl_launcher/data/)

转换数据管理相关代码：

- **replay_buffer.py**: 经验回放缓冲区
- **memory_efficient_replay_buffer.py**: 内存高效的回放缓冲区
- **dataset.py**: 数据集加载器
- **data_store.py**: 数据存储

优化策略：

- 使用 `torch.utils.data.DataLoader` 实现高效数据加载
- 应用 `pin_memory=True` 和 `num_workers` 进行异步数据传输
- 使用 `torch.multiprocessing` 实现并行采样
- 考虑使用 PyTorch 的 `DistributedSampler` 进行分布式训练

## 5. 公共工具模块转换 (serl_launcher/common/)

- **common.py**: 公共工具函数和类
  - `TrainState` 从 Flax 转换为 PyTorch 状态管理
  - `target_update` 使用 PyTorch 的参数复制
- **encoding.py**: 编码包装器
- **evaluation.py**: 评估工具
- **optimizers.py**: 优化器配置
  - `optax.adam` → `torch.optim.AdamW`
  - 学习率调度器转换
- **typing.py**: 类型定义更新

## 6. 环境包装器转换 (serl_launcher/wrappers/)

转换 Gym 环境包装器（保持接口兼容）：

- **serl_obs_wrappers.py**: 观察包装器
- **chunking.py**: 动作分块包装器
- **front_camera_wrapper.py**: 相机包装器
- 其他包装器确保与 PyTorch 张量兼容

## 7. 工具函数转换 (serl_launcher/utils/)

- **jax_utils.py**: 重命名为 `torch_utils.py`，转换所有 JAX 特定工具
- **train_utils.py**: 训练工具函数
- **timer_utils.py**: 保持不变
- **launcher.py**: 更新以支持 PyTorch

## 8. 机器人控制接口重构 (serl_robot_infra/)

移除 ROS 依赖，创建抽象接口：

### 8.1 创建抽象基类 (robot_servers/)

```python
# robot_servers/base_robot_server.py
class BaseRobotServer(ABC):
    @abstractmethod
    def move_to_joint_positions(self, positions): pass
    
    @abstractmethod
    def get_state(self): pass
    
    @abstractmethod
    def reset(self): pass

# robot_servers/base_gripper_server.py  
class BaseGripperServer(ABC):
    @abstractmethod
    def open(self): pass
    
    @abstractmethod
    def close(self): pass
```

### 8.2 更新现有服务器

- **franka_server.py**: 继承 `BaseRobotServer`，移除 `rospy` 导入
- **franka_gripper_server.py**: 继承 `BaseGripperServer`，移除 ROS 依赖
- **robotiq_gripper_server.py**: 同上
- 使用 Flask/FastAPI 或直接 Python SDK 替代 ROS 通信

### 8.3 更新环境代码

- **franka_env/envs/franka_env.py**: 更新以使用新的抽象接口
- 各个任务环境保持接口不变

## 9. 示例代码转换 (examples/)

转换 1 个代表性示例（SAC 状态输入）：

### 9.1 async_sac_state_sim  

- **async_sac_state_sim.py**: 完整的 PyTorch SAC 实现
- **run_actor.sh**, **run_learner.sh**: 更新启动脚本
- 包含完整的训练循环和评估逻辑

## 10. PyTorch 性能优化

应用关键优化技术以接近 JAX 性能：

### 10.1 编译优化

- 使用 `torch.compile(mode='max-autotune')` 编译关键函数
- 对训练循环应用 `torch.compile(fullgraph=True)`

### 10.2 内存优化

- 使用 `torch.cuda.amp.autocast()` 混合精度训练
- 应用梯度累积减少内存占用
- 使用 `torch.utils.checkpoint` 进行梯度检查点

### 10.3 计算优化

- 使用 CUDA graphs 减少启动开销（PyTorch 2.0+）
- 启用 `torch.backends.cudnn.benchmark = True`
- 使用 `torch.set_float32_matmul_precision('high')`
- 应用 Flash Attention（如果适用）

### 10.4 数据加载优化

- 使用 `DataLoader` 的 `persistent_workers=True`
- 优化批处理大小以最大化 GPU 利用率
- 使用 `prefetch_factor` 预取数据

## 11. 依赖更新

更新所有依赖文件：

### 11.1 serl_launcher/requirements.txt

移除：

- jax, jaxlib
- flax, optax, chex, distrax
- orbax-checkpoint

添加：

- torch>=2.1.0
- torchvision
- tensorboard（替代部分 TensorFlow 功能）

保留：

- gym, numpy, scipy
- wandb
- imageio, moviepy
- einops

### 11.2 serl_robot_infra（如需要）

- 移除 rospy 相关依赖
- 添加机器人 SDK（待定）

## 12. 文档和配置更新

- **README.md**: 更新安装说明，反映 PyTorch 依赖
- **docs/sim_quick_start.md**: 更新 SAC 示例代码
- 添加 PyTorch 优化使用指南
- 更新性能基准对比

## 实施注意事项

1. **测试策略**: 每个模块转换后进行单元测试，确保数值等价性
2. **渐进式迁移**: 按模块顺序转换，保持代码可运行状态
3. **性能基准**: 记录 JAX 版本的性能指标，转换后对比
4. **代码规范**: 遵循描述性变量名，类型注解函数签名
5. **向后兼容**: 保持 API 接口尽可能一致

### To-dos

- [ ] 转换神经网络模块 (networks/) - Flax 转 PyTorch
- [ ] 转换 RL 算法 (agents/) - SAC, DRQ, BC, VICE
- [ ] 转换视觉编码器 (vision/) - ResNet, MobileNet, 数据增强
- [ ] 转换数据处理模块 (data/) - replay buffer, dataset
- [ ] 转换公共工具和工具函数 (common/, utils/)
- [ ] 转换环境包装器 (wrappers/)
- [ ] 重构机器人接口，移除 ROS，创建抽象基类
- [ ] 转换示例代码 - async_drq_sim 和 async_sac_state_sim
- [ ] 应用 PyTorch 优化 - torch.compile, 混合精度, CUDA graphs
- [ ] 更新 requirements.txt - 移除 JAX，添加 PyTorch
- [ ] 更新文档和 README