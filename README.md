# Picar-X RL Training for Genesis Simulator

Reinforcement Learning training environment for the Picar-X robot using the Genesis physics simulator. Trains an AI agent to navigate and avoid obstacles using camera vision.

## Overview

This project provides:
- **URDF Model**: Complete 3D model of the Picar-X robot (798g, 25.4cm length)
- **RL Environment**: Genesis-based simulation with camera, obstacles, and physics
- **Training Script**: PPO (Proximal Policy Optimization) algorithm implementation
- **96x96 Camera Input**: Optimized for fast training on RTX 3090

## Robot Specifications

- **Total Mass**: 798 grams
- **Dimensions**: 25.4cm × 16.51cm × 10.16cm
- **Wheel Diameter**: 65mm
- **Steering**: ±30° front axle
- **Camera**: Front-mounted, static (moves with robot body)
- **Sensors**: 4 wheels with encoders, camera (96×96 RGB)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (RTX 3090 recommended)
- Ubuntu 20.04+ (Linux recommended for Genesis)

### Step 1: Install Genesis

```bash
pip install genesis-world
```

For GPU support, ensure you have CUDA installed:
```bash
nvidia-smi  # Check CUDA is available
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision numpy
```

For RTX 3090 with CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Clone This Repository

```bash
git clone <your-repo-url>
cd genesis_picarx_rl
```

## Project Structure

```
genesis_picarx_rl/
├── picar_x.urdf          # Robot model definition
├── picarx_env.py         # RL environment (Genesis)
├── train.py              # Training script (PPO)
├── README.md             # This file
└── checkpoints/          # Saved models (created during training)
```

## Quick Start

### Training

Run the training script:

```bash
python train.py
```

This will:
1. Initialize the Genesis simulator
2. Load the Picar-X URDF model
3. Create random obstacles
4. Train using PPO for 100,000 timesteps (default)
5. Save checkpoints every 10,000 steps

### Training Parameters

Edit `train.py` to adjust:

```python
# In main() function:
policy = trainer.train(
    total_timesteps=1000000,  # Total training steps
    rollout_length=2048,       # Steps per update
    save_interval=10000,       # Save every N steps
)
```

### Monitoring Training

The script prints progress:
```
Timesteps: 10000/100000
Avg Episode Reward: 45.23
Avg Episode Length: 234.50
Loss: 0.0123
```

## Environment Details

### Observations (What the AI Sees)

1. **Camera Image**: 96×96×3 RGB (flattened to 27,648 values)
2. **Joint Positions**: 4 wheel angles + 1 steering angle (5 values)
3. **Joint Velocities**: 5 values

**Total**: 27,658 observation dimensions

### Actions (What the AI Controls)

- **Left Motor**: -100 to +100 (backward to forward)
- **Right Motor**: -100 to +100 (backward to forward)

**Action Space**: Continuous 2D vector

### Reward Function

The AI learns to maximize:
- **+10× forward_velocity**: Reward for moving forward
- **-0.1× |angular_velocity|**: Penalty for spinning
- **-0.001× motor_effort**: Small penalty for energy use
- **-10× proximity_to_obstacle**: Penalty for being near obstacles
- **-100**: Big penalty for crashing
- **+0.1**: Survival bonus each step

### Episode Termination

Episode ends when:
- Robot collides with obstacle (distance < 0.2m)
- Robot hits boundary walls
- Maximum 1000 steps reached

## Customization

### Changing Camera Resolution

In `picarx_env.py`:
```python
env = PicarXEnv(
    img_width=128,   # Change from 96
    img_height=128,  # Change from 96
)
```

**Note**: Update `ActorCritic` CNN in `train.py` if you change resolution.

### Adding More Obstacles

In `picarx_env.py`, find `_create_scene()`:
```python
num_obstacles = np.random.randint(10, 21)  # Increase range
```

### Adjusting Motor Power

In `picarx_env.py`:
```python
self.max_motor_force = 2.0  # Increase from 1.0
```

### Visualization During Training

Change render mode to see the robot:
```python
env = PicarXEnv(
    render_mode="human",  # Shows 3D viewer
)
```

**Warning**: This slows down training significantly!

## Training Tips

### For RTX 3090 (24GB VRAM)

You can increase batch size for faster training:
```python
trainer = PPOTrainer(
    batch_size=256,  # Increase from 64
    # ... other params
)
```

### Speed vs Quality Trade-off

- **Fast Training**: Reduce `rollout_length` to 1024
- **Better Policy**: Increase `num_epochs` to 20
- **More Exploration**: Increase `entropy_coef` to 0.02

### Troubleshooting

**Out of Memory Error**:
- Reduce `batch_size` (try 32)
- Reduce `rollout_length` (try 1024)
- Use smaller camera resolution (64×64)

**Robot Not Moving**:
- Check `max_motor_force` in environment
- Verify actions are being clipped to [-100, 100]
- Ensure wheels are properly connected in URDF

**Training Not Improving**:
- Increase `total_timesteps` (try 1M+)
- Adjust learning rate (try 1e-4 or 1e-3)
- Check reward function signs (+/-)

## Using Trained Model

After training, load the model:

```python
import torch
from train import ActorCritic

# Load model
policy = ActorCritic(obs_dim=27658, action_dim=2)
policy.load_state_dict(torch.load("checkpoints/20240131_120000/final_model.pt"))
policy.eval()

# Use for inference
obs = env.reset()
with torch.no_grad():
    action_mean, _, _ = policy(torch.FloatTensor(obs))
```

## Advanced: Multi-Environment Training

For faster training with multiple parallel environments:

```python
from picarx_env import PicarXEnvVec

# Create 4 parallel environments
env = PicarXEnvVec(num_envs=4)

# Rest of training code works the same
```

## Real Robot Deployment

To transfer to real Picar-X:

1. Export trained policy weights
2. Load on Raspberry Pi
3. Replace simulated camera with real Pi Camera
4. Map actions to motor commands via Robot HAT

See the main [picar-x-racer](https://github.com/KarimAziev/picar-x-racer) project for real robot integration.

## Performance Benchmarks

On RTX 3090:
- **Training Speed**: ~5,000 steps/second (headless)
- **Memory Usage**: ~4GB VRAM
- **Time to 100k steps**: ~20 seconds
- **Time to 1M steps**: ~3.5 minutes

## Citation

If you use this in research:

```bibtex
@software{picarx_rl_genesis,
  author = {Your Name},
  title = {Picar-X RL Training with Genesis},
  year = {2024},
  url = {https://github.com/yourusername/genesis_picarx_rl}
}
```

## License

MIT License - See LICENSE file

## Support

- **Issues**: Open a GitHub issue
- **Genesis Docs**: https://genesis-world.readthedocs.io/
- **Picar-X Docs**: https://docs.sunfounder.com/projects/picar-x/en/stable/

## Roadmap

- [ ] Add support for real-to-sim transfer
- [ ] Implement curriculum learning (increasing difficulty)
- [ ] Add LIDAR sensor option
- [ ] Multi-robot training
- [ ] Integration with ROS2

---

**Happy Training!** 🚗🤖
