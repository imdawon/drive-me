import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
import cv2
import os

# --- 1. GENERATE ROBOT ASSET (URDF) ---
def generate_urdf():
    urdf_content = """<?xml version="1.0"?>
<robot name="picar">
  <link name="base_link">
    <visual>
      <geometry><box size="0.25 0.12 0.06"/></geometry>
      <material name="blue"><color rgba="0 0 1 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.25 0.12 0.06"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="wheel_fl">
    <visual>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
      <material name="black"><color rgba="0 0 0 1"/></material>
    </visual>
    <collision>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint_fl" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fl"/>
    <origin xyz="0.1 0.08 -0.02"/>
    <axis xyz="0 1 0"/>
  </joint>
  <link name="wheel_fr">
    <visual>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </visual>
    <collision>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint_fr" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fr"/>
    <origin xyz="0.1 -0.08 -0.02"/>
    <axis xyz="0 1 0"/>
  </joint>
  <link name="wheel_rl">
    <visual>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </visual>
    <collision>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint_rl" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_rl"/>
    <origin xyz="-0.1 0.08 -0.02"/>
    <axis xyz="0 1 0"/>
  </joint>
  <link name="wheel_rr">
    <visual>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </visual>
    <collision>
      <geometry><cylinder length="0.02" radius="0.035"/></geometry>
      <origin rpy="1.5707 0 0" xyz="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint_rr" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_rr"/>
    <origin xyz="-0.1 -0.08 -0.02"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""
    with open("picar.urdf", "w") as f:
        f.write(urdf_content)
    return "picar.urdf"

# --- 2. SETUP GENESIS ---
gs.init(backend=gs.gpu, logging_level="warning")

def create_scene():
    urdf_path = generate_urdf()
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        show_viewer=False,
    )
    
    plane = scene.add_entity(gs.morphs.Plane())
    
    picar = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0, 0, 0.2),
            fixed=False
        ),
    )
    
    cam = scene.add_camera(res=(640, 480), pos=(8.0, -8.0, 5.0), lookat=(4.0, 0.0, 0.0), fov=60)
    
    scene.build(n_envs=2048)
    return scene, picar, cam

# --- 3. MODEL ---
class SimplePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.act_scale = 30.0
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, x):
        logits = self.net(x)
        mean = torch.tanh(logits) * self.act_scale
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

# --- 4. TRAIN ---
def train():
    scene, picar, cam = create_scene()
    
    j_names = ["joint_fl", "joint_fr", "joint_rl", "joint_rr"]
    try:
        motor_indices = [picar.get_joint(name).dofs_idx_local[0] for name in j_names]
    except:
        motor_indices = [6, 7, 8, 9] 

    print(f"Motor Indices detected at: {motor_indices}")
    
    target_pos = torch.tensor([5.0, 0.0, 0.0], device="cuda")
    
    policy = SimplePolicy(obs_dim=13, act_dim=4).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    full_cmds = torch.zeros((2048, 10), device="cuda")
    
    print("Training Started (Short Run: 5 Epochs)...")
    
    for epoch in range(5):
        scene.reset()
        
        log_probs_list = []
        entropies_list = []
        rewards_list = []
        
        for step in range(500):
            pos = picar.get_pos()
            quat = picar.get_quat()
            lin_vel = picar.get_vel()
            ang_vel = picar.get_ang()
            
            rel_pos = pos - target_pos
            rel_xy = pos[:, :2] - target_pos[:2]
            old_dist = torch.norm(rel_xy, dim=1)
            
            obs = torch.cat([rel_pos / 5.0, quat, lin_vel / 5.0, ang_vel / 10.0], dim=1)
            
            mean, std = policy(obs)
            distri = D.Normal(mean, std)
            action = distri.rsample()
            log_prob = distri.log_prob(action).sum(dim=-1)
            entropy = distri.entropy().sum(dim=-1)
            
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)
            
            full_cmds.fill_(0)
            for i, idx in enumerate(motor_indices):
                full_cmds[:, idx] = action[:, i]
            
            picar.control_dofs_velocity(full_cmds)
            scene.step()
            
            new_pos = picar.get_pos()
            new_rel_xy = new_pos[:, :2] - target_pos[:2]
            new_dist = torch.norm(new_rel_xy, dim=1)
            
            new_quat = picar.get_quat()
            yaw = torch.atan2(2 * (new_quat[:,0] * new_quat[:,3] + new_quat[:,1] * new_quat[:,2]),
                              1 - 2 * (new_quat[:,2]**2 + new_quat[:,3]**2))
            desired_yaw = torch.atan2(new_rel_xy[:,1], new_rel_xy[:,0])
            diff = desired_yaw - yaw
            heading_error = torch.atan2(torch.sin(diff), torch.cos(diff))
            heading_bonus = torch.cos(heading_error)
            
            reward = old_dist - new_dist
            reward += 0.05 * heading_bonus
            reward -= (action ** 2).mean(dim=1) * 0.0001
            
            rewards_list.append(reward)
        
        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropies_list, dim=1)
        rewards = torch.stack(rewards_list, dim=1)
        returns = rewards.sum(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        entropy_bonus = entropies.mean(dim=1)
        
        loss = -((log_probs.sum(dim=1) * returns) + 0.01 * entropy_bonus).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        final_dist = new_dist.mean().item()
        print(f"Epoch {epoch}: Avg Final Distance: {final_dist:.2f}")
    
    torch.save(policy.state_dict(), "picar_policy.pth")
    
    print("Saving video...")
    scene.reset()
    out = cv2.VideoWriter('picar_training.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    
    for i in range(300):
        pos = picar.get_pos()
        quat = picar.get_quat()
        lin_vel = picar.get_vel()
        ang_vel = picar.get_ang()
        
        rel_pos = pos - target_pos
        obs = torch.cat([rel_pos / 5.0, quat, lin_vel / 5.0, ang_vel / 10.0], dim=1)
        
        mean, _ = policy(obs)
        actions = mean
        
        full_cmds.fill_(0)
        for j, idx in enumerate(motor_indices):
            full_cmds[:, idx] = actions[:, j]
        picar.control_dofs_velocity(full_cmds)
        
        scene.step()
        
        # --- FIX 1: Update Camera Pose ---
        car_pos = pos[0].cpu().numpy()
        cam.set_pose(
            pos=(car_pos[0] - 2.0, car_pos[1] - 2.0, 1.5), 
            lookat=(car_pos[0], car_pos[1], 0.0)
        )
        
        # --- FIX 2: Render correctly (No 'envs' arg) ---
        rgb, _, _, _ = cam.render()
        
        # --- FIX 3: Robust Image Handling (Numpy check) ---
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
            
        # If output is (Batch, H, W, C), take first element. 
        # If output is (H, W, C), take it directly.
        if rgb.ndim == 4:
            frame = rgb[0]
        else:
            frame = rgb
            
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()
    print("Done! Video saved to 'picar_training.mp4'")

if __name__ == "__main__":
    train()
