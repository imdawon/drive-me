import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
import cv2
import os

if torch.cuda.is_available():
    device = "cuda"
    genesis_backend = gs.gpu
elif torch.backends.mps.is_available():
    device = "mps"
    genesis_backend = gs.cpu
else:
    device = "cpu"
    genesis_backend = gs.cpu
gs.init(backend=genesis_backend, logging_level="warning")

def generate_urdf():
    urdf_content = """
<?xml version="1.0"?>
<robot name="picar">
  <link name="base_link">
    <visual>
      <geometry><box size="0.26 0.17 0.06"/></geometry>
      <material name="gray"><color rgba="0.7 0.7 0.7 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.26 0.17 0.06"/></geometry>
    </collision>
    <inertial>
      <mass value="1.8"/>
      <inertia ixx="0.08" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.08"/>
    </inertial>
  </link>
  <link name="wheel_fl">
    <visual>
      <geometry><cylinder length="0.03" radius="0.032"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
      <material name="white"><color rgba="1 1 1 1"/></material>
    </visual>
    <collision>
      <geometry><cylinder length="0.03" radius="0.032"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint_fl" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fl"/>
    <origin xyz="0.13 0.085 -0.032"/>
    <axis xyz="0 1 0"/>
    <limit effort="20.0" velocity="40.0"/>
  </joint>
  <link name="wheel_fr">
    <visual>
      <geometry><cylinder length="0.03" radius="0.032"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
      <material name="white"><color rgba="1 1 1 1"/></material>
    </visual>
    <collision>
      <geometry><cylinder length="0.03" radius="0.032"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint_fr" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fr"/>
    <origin xyz="0.13 -0.085 -0.032"/>
    <axis xyz="0 1 0"/>
    <limit effort="20.0" velocity="40.0"/>
  </joint>
  <link name="wheel_rl">
    <visual>
      <geometry><cylinder length="0.04" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <geometry><cylinder length="0.04" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.12"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>
  <joint name="joint_rl" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_rl"/>
    <origin xyz="-0.13 0.085 -0.035"/>
    <axis xyz="0 1 0"/>
    <limit effort="20.0" velocity="40.0"/>
  </joint>
  <link name="wheel_rr">
    <visual>
      <geometry><cylinder length="0.04" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <geometry><cylinder length="0.04" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.12"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>
  <joint name="joint_rr" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_rr"/>
    <origin xyz="-0.13 -0.085 -0.035"/>
    <axis xyz="0 1 0"/>
    <limit effort="20.0" velocity="40.0"/>
  </joint>
  <link name="camera_link">
    <visual>
      <geometry><box size="0.04 0.06 0.04"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <material name="black"><color rgba="0.2 0.2 0.2 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.04 0.06 0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.14 0 0.10" rpy="0 -0.17 0"/>
  </joint>
</robot>
"""
    with open("picar.urdf", "w") as f:
        f.write(urdf_content)
    return "picar.urdf"

def create_scene():
    urdf_path = generate_urdf()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=4),
        show_viewer=True,
    )
    plane = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.2))
    picar = scene.add_entity(
        gs.morphs.URDF(file=urdf_path, pos=(0, 0, 0.035), fixed=False),
    )
    target = scene.add_entity(
        gs.morphs.Box(size=(0.406, 0.254, 0.305)),
        pos=(3.0, 0.0, 0.1525),
        fixed=True,
    )
    target.set_material(gs.materials.Visual(color=(0.65, 0.45, 0.30, 1.0)), gs.materials.Rigid(friction=1.0))
    viewer_cam = scene.add_camera(res=(640, 480), pos=(8.0, -8.0, 5.0), lookat=(0.0, 0.0, 0.0), fov=60)
    robot_cam = scene.add_camera(res=(96, 96), fov=70, near=0.01, far=20.0)
    scene.build(n_envs=2048)
    return scene, picar, target, viewer_cam, robot_cam

def update_robot_camera(robot_cam, picar):
    link_pos = picar.get_link_pos("camera_link")
    link_quat = picar.get_link_quat("camera_link")
    forward_local = torch.tensor([[1.0, 0.0, 0.0]], device="cpu").repeat(2048, 1)
    forward = gs.math.quat_apply(link_quat, forward_local)
    lookat = link_pos + forward * 3.0
    robot_cam.set_pose(pos=link_pos, lookat=lookat)

class VisionPolicy(nn.Module):
    def __init__(self, act_dim=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(512, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.act_scale = 21.0

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        logits = self.mean_head(features)
        mean = torch.tanh(logits) * self.act_scale
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

def train():
    scene, picar, target, viewer_cam, robot_cam = create_scene()
    j_names = ["joint_rl", "joint_rr"]
    motor_indices = [picar.get_joint(name).dofs_idx_local[0] for name in j_names]

    policy = VisionPolicy(act_dim=2).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    full_cmds = torch.zeros((2048, 20), device="cpu")

    max_steps = 500
    contact_threshold = 0.45

    for epoch in range(40):
        scene.reset()

        robot_angles = torch.rand(2048, device="cpu") * 2 * np.pi
        robot_quat = gs.math.euler_to_quat(torch.stack([torch.zeros(2048, device="cpu"), torch.zeros(2048, device="cpu"), robot_angles], dim=1))
        picar.set_quat(robot_quat)

        target_angles = torch.rand(2048, device="cpu") * 2 * np.pi
        target_dists = 0.6 + torch.rand(2048, device="cpu") * 2.4
        target_x = target_dists * torch.cos(target_angles)
        target_y = target_dists * torch.sin(target_angles)
        target_pos = torch.stack([target_x, target_y, torch.full((2048,), 0.1525, device="cpu")], dim=1)
        target.set_pos(target_pos)

        pos = picar.get_pos()
        previous_dist = torch.norm(pos[:, :2] - target_pos[:, :2], dim=1)

        update_robot_camera(robot_cam, picar)

        done = torch.zeros(2048, dtype=torch.bool, device="cpu")

        log_probs_list = []
        entropies_list = []
        rewards_list = []

        for step in range(max_steps):
            rgb = robot_cam.render()
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            obs_gpu = rgb.to(device)

            mean, std = policy(obs_gpu)
            distri = D.Normal(mean, std)
            action_gpu = distri.rsample()

            log_prob = distri.log_prob(action_gpu).sum(dim=-1)
            entropy = distri.entropy().sum(dim=-1)

            log_probs_list.append(log_prob)
            entropies_list.append(entropy)

            action_cpu = action_gpu.cpu()
            action_cpu[done] = 0.0

            full_cmds.fill_(0)
            full_cmds[:, motor_indices[0]] = action_cpu[:, 0]
            full_cmds[:, motor_indices[1]] = action_cpu[:, 1]

            picar.control_dofs_velocity(full_cmds)
            scene.step()

            update_robot_camera(robot_cam, picar)

            new_pos = picar.get_pos()
            new_dist = torch.norm(new_pos[:, :2] - target_pos[:, :2], dim=1)

            new_quat = picar.get_quat()
            yaw = torch.atan2(2 * (new_quat[:,0] * new_quat[:,3] + new_quat[:,1] * new_quat[:,2]),
                              1 - 2 * (new_quat[:,2]**2 + new_quat[:,3]**2))
            desired_yaw = torch.atan2(new_pos[:,1] - target_pos[:,1], new_pos[:,0] - target_pos[:,0])
            diff = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            heading_bonus = torch.cos(diff)

            reward = (previous_dist - new_dist) * 10.0
            reward += heading_bonus
            reward -= (action_cpu ** 2).mean(dim=1) * 0.0001
            reward -= 0.02

            contact = new_dist < contact_threshold
            new_contact = contact & ~done
            reward[new_contact] += 200.0

            done = done | contact

            previous_dist = new_dist.clone()

            rewards_list.append(reward.to(device))

        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropies_list, dim=1)
        rewards = torch.stack(rewards_list, dim=1)
        returns = rewards.sum(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -((log_probs.sum(dim=1) * returns.detach()).mean() + 0.01 * entropies.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        success_rate = done.float().mean().item()
        avg_dist = new_dist.mean().item()
        print(f"Epoch {epoch}: Success {success_rate:.3f} Final Dist {avg_dist:.2f}")

    torch.save(policy.state_dict(), "picar_vision_policy.pth")

    scene.reset()
    target.set_pos(torch.tensor([[3.0, 0.0, 0.1525]] * 2048, device="cpu"))
    update_robot_camera(robot_cam, picar)

    out = cv2.VideoWriter('picar_vision_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, (640, 480))
    done_demo = torch.zeros(2048, dtype=torch.bool, device="cpu")

    for i in range(max_steps):
        rgb = robot_cam.render()
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        obs_gpu = rgb[0:1].to(device)

        mean, _ = policy(obs_gpu)
        actions_cpu = mean.cpu()
        actions_cpu[done_demo] = 0.0

        full_cmds.fill_(0)
        full_cmds[:, motor_indices[0]] = actions_cpu[:, 0]
        full_cmds[:, motor_indices[1]] = actions_cpu[:, 1]

        picar.control_dofs_velocity(full_cmds)
        scene.step()

        update_robot_camera(robot_cam, picar)

        car_pos = picar.get_pos()[0].cpu().numpy()
        viewer_cam.set_pose(pos=(car_pos[0]-4, car_pos[1]-4, 3.0), lookat=(car_pos[0], car_pos[1], 0.0))

        frame_rgb, _, _, _ = viewer_cam.render()
        if frame_rgb.ndim == 4:
            frame_rgb = frame_rgb[0]
        frame = (frame_rgb.cpu().numpy() * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

        new_dist_demo = torch.norm(picar.get_pos()[:, :2] - target.get_pos()[:, :2], dim=1)
        done_demo = new_dist_demo < contact_threshold

    out.release()
    print("Demo video saved")

if __name__ == "__main__":
    train()