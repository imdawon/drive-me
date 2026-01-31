"""
Picar-X RL Environment for Genesis Simulator
Navigation and obstacle avoidance training environment.
"""

import numpy as np
import genesis as gs
from typing import Tuple, Dict, Any
import torch


class PicarXEnv:
    """
    Reinforcement Learning Environment for Picar-X robot.

    Task: Navigate through environment while avoiding obstacles.

    Observations:
        - Camera image (96x96 RGB)
        - Joint positions (wheel angles)
        - Motor velocities

    Actions (Continuous):
        - Left motor speed: [-100, 100]
        - Right motor speed: [-100, 100]

    Rewards:
        - Positive: Moving forward, avoiding obstacles
        - Negative: Crashing, staying still
    """

    def __init__(
        self,
        num_envs: int = 1,
        render_mode: str = None,
        max_steps: int = 1000,
        img_width: int = 96,
        img_height: int = 96,
    ):
        """
        Initialize the environment.

        Args:
            num_envs: Number of parallel environments (for vectorized training)
            render_mode: 'human' for visualization, None for headless
            max_steps: Maximum steps per episode
            img_width: Camera image width
            img_height: Camera image height
        """
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.img_width = img_width
        self.img_height = img_height
        self.current_step = 0

        # Action space: [left_motor, right_motor] in range [-100, 100]
        self.action_dim = 2
        self.action_low = -100.0
        self.action_high = 100.0

        # Observation space: flattened image + joint positions + velocities
        # Image: 96x96x3 = 27648
        # Joint positions: 4 wheel joints + 1 steering joint = 5
        # Joint velocities: 5
        self.obs_dim = (img_width * img_height * 3) + 5 + 5

        # Initialize Genesis
        gs.init(backend=gs.gpu)

        # Create scene
        self.scene = self._create_scene()

        # Build scene
        self.scene.build()

        # Get robot entity
        self.robot = self.scene.entities[1]  # Index 1 is the robot (0 is plane)

        # Camera setup
        self.camera = self.scene.add_camera(
            pos=(0, 0.1, 0.12),  # Mounted on robot front
            lookat=(0, 1.0, 0.12),
            up=(0, 0, 1),
            fov=60,
            res=(img_width, img_height),
        )

        # Joint indices
        self.wheel_joints = [
            self.robot.get_joint("front_left_wheel_joint"),
            self.robot.get_joint("front_right_wheel_joint"),
            self.robot.get_joint("back_left_wheel_joint"),
            self.robot.get_joint("back_right_wheel_joint"),
        ]
        self.steering_joint = self.robot.get_joint("front_axle_steering_joint")

        # Motor parameters
        self.max_motor_force = 1.0
        self.wheel_radius = 0.0325

        # Episode tracking
        self.episode_rewards = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs)

    def _create_scene(self):
        """Create the simulation scene with obstacles."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.02,  # 50 Hz simulation
                substeps=10,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2, 2, 2),
                camera_lookat=(0, 0, 0),
                camera_fov=45,
                max_FPS=60,
            )
            if self.render_mode == "human"
            else None,
            show_viewer=self.render_mode == "human",
        )

        # Add ground plane
        plane = scene.add_entity(
            gs.morphs.Plane(),
        )

        # Add robot from URDF
        robot = scene.add_entity(
            gs.morphs.URDF(
                file="picar_x.urdf",
                pos=(0, 0, 0.1),
                fixed=False,
            ),
        )

        # Add obstacles (random boxes and cylinders)
        self.obstacles = []
        np.random.seed(42)  # For reproducibility

        # Add 5-10 random obstacles
        num_obstacles = np.random.randint(5, 11)
        for i in range(num_obstacles):
            # Random position (avoid spawn area)
            x = np.random.uniform(-3, 3)
            y = (
                np.random.uniform(1, 5)
                if np.random.rand() > 0.5
                else np.random.uniform(-5, -1)
            )

            # Random obstacle type
            if np.random.rand() > 0.5:
                # Box obstacle
                size = np.random.uniform(0.1, 0.3)
                obstacle = scene.add_entity(
                    gs.morphs.Box(
                        size=(size, size, size),
                        pos=(x, y, size / 2),
                    ),
                )
            else:
                # Cylinder obstacle
                radius = np.random.uniform(0.05, 0.15)
                height = np.random.uniform(0.1, 0.4)
                obstacle = scene.add_entity(
                    gs.morphs.Cylinder(
                        radius=radius,
                        height=height,
                        pos=(x, y, height / 2),
                    ),
                )

            self.obstacles.append(obstacle)

        # Add boundary walls
        wall_height = 0.5
        wall_thickness = 0.1
        arena_size = 6.0

        # Four walls
        walls = [
            (
                (-arena_size, 0, wall_height / 2),
                (wall_thickness, arena_size, wall_height),
            ),  # Left
            (
                (arena_size, 0, wall_height / 2),
                (wall_thickness, arena_size, wall_height),
            ),  # Right
            (
                (0, arena_size, wall_height / 2),
                (arena_size, wall_thickness, wall_height),
            ),  # Front
            (
                (0, -arena_size, wall_height / 2),
                (arena_size, wall_thickness, wall_height),
            ),  # Back
        ]

        for pos, size in walls:
            scene.add_entity(
                gs.morphs.Box(
                    size=size,
                    pos=pos,
                ),
            )

        return scene

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information
        """
        self.current_step = 0

        # Reset robot position (random spawn in safe area)
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        yaw = np.random.uniform(-np.pi, np.pi)

        self.robot.set_pos(np.array([x, y, 0.1]))
        self.robot.set_quat(
            np.array(
                [
                    np.cos(yaw / 2),
                    0,
                    0,
                    np.sin(yaw / 2),  # Quaternion from yaw
                ]
            )
        )

        # Reset velocities
        self.robot.set_vel(np.zeros(3))
        self.robot.set_ang_vel(np.zeros(3))

        # Reset joint positions and velocities
        for joint in self.wheel_joints:
            joint.set_pos(0)
            joint.set_vel(0)
        self.steering_joint.set_pos(0)
        self.steering_joint.set_vel(0)

        # Get initial observation
        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.

        Args:
            action: [left_motor, right_motor] in range [-100, 100]

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (collision, etc.)
            truncated: Whether episode was cut short (max steps)
            info: Additional information
        """
        self.current_step += 1

        # Clip actions to valid range
        action = np.clip(action, self.action_low, self.action_high)
        left_motor, right_motor = action

        # Convert motor commands to wheel torques
        # Simple differential drive model
        left_torque = (left_motor / 100.0) * self.max_motor_force
        right_torque = (right_motor / 100.0) * self.max_motor_force

        # Apply torques to wheels
        self.wheel_joints[0].set_vel(left_torque * 10)  # Front left
        self.wheel_joints[1].set_vel(right_torque * 10)  # Front right
        self.wheel_joints[2].set_vel(left_torque * 10)  # Back left
        self.wheel_joints[3].set_vel(right_torque * 10)  # Back right

        # Simulate one step
        self.scene.step()

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward, terminated = self._calculate_reward(action)

        # Check truncation (max steps)
        truncated = self.current_step >= self.max_steps

        # Additional info
        info = {
            "step": self.current_step,
            "motor_left": left_motor,
            "motor_right": right_motor,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            observation: Concatenated [flattened_image, joint_positions, joint_velocities]
        """
        # Update camera position to follow robot
        robot_pos = self.robot.get_pos()
        robot_quat = self.robot.get_quat()

        # Camera offset from robot center (front-mounted)
        camera_offset = np.array([0, 0.1, 0.12])

        # Transform offset by robot rotation
        # Simple rotation around Z axis
        yaw = 2 * np.arctan2(robot_quat[3], robot_quat[0])
        rotated_offset = np.array(
            [
                camera_offset[0] * np.cos(yaw) - camera_offset[1] * np.sin(yaw),
                camera_offset[0] * np.sin(yaw) + camera_offset[1] * np.cos(yaw),
                camera_offset[2],
            ]
        )

        camera_pos = robot_pos + rotated_offset
        lookat_pos = camera_pos + np.array([0.5 * np.sin(yaw), 0.5 * np.cos(yaw), 0])

        self.camera.set_pose(pos=camera_pos, lookat=lookat_pos)

        # Capture image
        img = self.camera.render()

        # Flatten image
        img_flat = img.reshape(-1) / 255.0  # Normalize to [0, 1]

        # Get joint positions and velocities
        joint_positions = []
        joint_velocities = []

        for joint in self.wheel_joints:
            joint_positions.append(joint.get_pos())
            joint_velocities.append(joint.get_vel())

        joint_positions.append(self.steering_joint.get_pos())
        joint_velocities.append(self.steering_joint.get_vel())

        joint_positions = np.array(joint_positions)
        joint_velocities = np.array(joint_velocities)

        # Concatenate all observations
        obs = np.concatenate([img_flat, joint_positions, joint_velocities])

        return obs.astype(np.float32)

    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, bool]:
        """
        Calculate reward for current state.

        Args:
            action: The action taken

        Returns:
            reward: Calculated reward
            terminated: Whether episode should end
        """
        reward = 0.0
        terminated = False

        # Get robot state
        robot_pos = self.robot.get_pos()
        robot_vel = self.robot.get_vel()

        # Reward for forward movement
        forward_vel = robot_vel[1]  # Y-axis is forward
        reward += forward_vel * 10.0  # Scale factor

        # Small penalty for spinning in place (encourage forward motion)
        angular_vel = self.robot.get_ang_vel()[2]  # Z-axis rotation
        reward -= abs(angular_vel) * 0.1

        # Penalty for large actions (energy efficiency)
        left_motor, right_motor = action
        reward -= (abs(left_motor) + abs(right_motor)) * 0.001

        # Check for collisions (simple distance-based check)
        for obstacle in self.obstacles:
            obstacle_pos = obstacle.get_pos()
            distance = np.linalg.norm(robot_pos[:2] - obstacle_pos[:2])

            if distance < 0.2:  # Collision threshold
                reward -= 100.0  # Big penalty for collision
                terminated = True
                break
            elif distance < 0.5:  # Getting close
                reward -= (
                    0.5 - distance
                ) * 10.0  # Small penalty for being near obstacles

        # Check boundary collision
        if abs(robot_pos[0]) > 5.5 or abs(robot_pos[1]) > 5.5:
            reward -= 100.0
            terminated = True

        # Small survival reward
        reward += 0.1

        return reward, terminated

    def close(self):
        """Clean up resources."""
        gs.destroy()


# For compatibility with standard RL libraries
class PicarXEnvVec:
    """Vectorized version for parallel training."""

    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [PicarXEnv(**kwargs) for _ in range(num_envs)]

    def reset(self):
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(self, actions):
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()
