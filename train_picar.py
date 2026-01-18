import genesis as gs
import genesis.utils.geom as gu
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
import cv2
import os

# --- Backend Setup ---
# Check explicitly for cuda/mps to set backend, but handle errors gracefully
try:
    if torch.cuda.is_available():
        device = "cuda"
        genesis_backend = gs.gpu
    elif torch.backends.mps.is_available():
        device = "mps"
        genesis_backend = gs.cpu
    else:
        device = "cpu"
        genesis_backend = gs.cpu
except:
    device = "cpu"
    genesis_backend = gs.cpu

print(f"Using device: {device}, Genesis backend: {genesis_backend}")
gs.init(backend=genesis_backend, logging_level="warning")

# --- Math Helpers ---
def euler_to_quat_tensor(euler):
    """Batched tensor version for the training loop"""
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)

def get_camera_relative_quat():
    """Calculates the single relative quaternion for the camera attachment (CPU/Numpy)"""
    # URDF RPY: 0, -0.17, 0
    roll, pitch, yaw = 0.0, -0.17, 0.0
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)

# --- URDF Generation ---
def generate_urdf():
    urdf_content = """<?xml version="1.0"?>
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
</robot>"""
    with open("picar.urdf", "w") as f:
        f.write(urdf_content)
    return "picar.urdf"

# --- Scene Creation ---
def create_scene():
    urdf_path = generate_urdf()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=4),
        show_viewer=False,
    )
    plane = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.2))
    picar = scene.add_entity(
        gs.morphs.URDF(file=urdf_path, pos=(0, 0, 0.035), fixed=False),
    )
    target = scene.add_entity(
        gs.morphs.Box(size=(0.406, 0.254, 0.305), fixed=True),
        material=gs.materials.Rigid(friction=1.0)
    )
    viewer_cam = scene.add_camera(res=(640, 480), pos=(8.0, -8.0, 5.0), lookat=(0.0, 0.0, 0.0), fov=60)
    
    # Initialize with dummies
    robot_cam = scene.add_camera(
        res=(96, 96),
        pos=(0.0, 0.0, 0.0), 
        lookat=(1.0, 0.0, 0.0), 
        fov=70,
        near=0.01,
        far=20.0
    )

    # Attach with matrix transform using genesis.utils.geom
    rel_pos = np.array([0.14, 0.0, 0.10])
    rel_quat = np.array(get_camera_relative_quat())
    rel_transform = gu.trans_quat_to_T(rel_pos, rel_quat)

    robot_cam.attach(
        picar.get_link("base_link"),
        rel_transform
    )

    # Build scene with 2048 environments
    scene.build(n_envs=2048)
    return scene, picar, target, viewer_cam, robot_cam

# --- Policy and Training ---
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
        # x expected shape: (Batch, Height, Width, Channels) -> (B, H, W, C)
        # Permute to (Batch, Channels, Height, Width) -> (B, C, H, W)
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
    
    n_envs = 2048
    full_cmds = torch.zeros((n_envs, 20), device=device)
    max_steps = 500
    contact_threshold = 0.45

    for epoch in range(40):
        scene.reset()
        
        robot_angles = torch.rand(n_envs, device=device) * 2 * np.pi
        euler_input = torch.stack([torch.zeros(n_envs, device=device), torch.zeros(n_envs, device=device), robot_angles], dim=1)
        robot_quat = euler_to_quat_tensor(euler_input)
        picar.set_quat(robot_quat)
        
        target_angles = torch.rand(n_envs, device=device) * 2 * np.pi
        target_dists = 0.6 + torch.rand(n_envs, device=device) * 2.4
        target_x = target_dists * torch.cos(target_angles)
        target_y = target_dists * torch.sin(target_angles)
        target_pos = torch.stack([target_x, target_y, torch.full((n_envs,), 0.1525, device=device)], dim=1)
        target.set_pos(target_pos)
        
        pos = picar.get_pos()
        previous_dist = torch.norm(pos[:, :2] - target_pos[:, :2], dim=1)
        
        done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        log_probs_list = []
        entropies_list = []
        rewards_list = []
        
        for step in range(max_steps):
            if done.all():
                break
            
            # Rendering: returns numpy array
            rgb, _, _, _ = robot_cam.render()
            
            # --- CRITICAL FIX FOR DIMENSION MISMATCH ---
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            
            obs = torch.from_numpy(rgb).float().to(device)
            
            # Handle cases where Genesis returns single image (H, W, C) due to driver failure
            # or (N, H, W) missing channels
            if obs.dim() == 3:
                # Case 1: (H, W, C) -> Single image returned
                if obs.shape[-1] == 3:
                    # Expand to (B, H, W, C)
                    obs = obs.unsqueeze(0).expand(n_envs, -1, -1, -1)
                # Case 2: (B, H, W) -> Batched but grayscale/missing channel dim
                else:
                    obs = obs.unsqueeze(-1).expand(-1, -1, -1, 3) # Force 3 channels if missing
            elif obs.dim() == 4:
                # Correct shape (B, H, W, C)
                pass
            else:
                 raise RuntimeError(f"Unexpected observation shape: {obs.shape}")

            # -------------------------------------------

            mean, std = policy(obs)
            distri = D.Normal(mean, std)
            action = distri.rsample()
            log_prob = distri.log_prob(action).sum(dim=-1)
            entropy = distri.entropy().sum(dim=-1)
            
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)
            
            action_control = action.clone()
            action_control[done] = 0.0
            
            full_cmds.fill_(0)
            full_cmds[:, motor_indices[0]] = action_control[:, 0]
            full_cmds[:, motor_indices[1]] = action_control[:, 1]
            picar.control_dofs_velocity(full_cmds)
            
            scene.step()
            
            new_pos = picar.get_pos()
            new_dist = torch.norm(new_pos[:, :2] - target_pos[:, :2], dim=1)
            new_quat = picar.get_quat()
            
            yaw = torch.atan2(2 * (new_quat[:,0] * new_quat[:,3] + new_quat[:,1] * new_quat[:,2]), 
                              1 - 2 * (new_quat[:,2]**2 + new_quat[:,3]**2))
            
            desired_yaw = torch.atan2(target_pos[:,1] - new_pos[:,1], target_pos[:,0] - new_pos[:,0])
            diff = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            heading_bonus = torch.cos(diff)
            
            reward = (previous_dist - new_dist) * 10.0
            reward += heading_bonus
            reward -= (action ** 2).mean(dim=1) * 0.0001
            reward -= 0.02
            
            contact = new_dist < contact_threshold
            new_contact = contact & ~done
            reward[new_contact] += 200.0
            done = done | contact
            previous_dist = new_dist.clone()
            
            rewards_list.append(reward)
        
        # Stack lists. Check if empty to avoid crash if done immediately
        if len(log_probs_list) > 0:
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
        avg_dist = previous_dist.mean().item() # Use previous_dist as it holds last known dists
        print(f"Epoch {epoch}: Success {success_rate:.3f} Final Dist {avg_dist:.2f}")

    torch.save(policy.state_dict(), "picar_vision_policy.pth")
    
    # Demo Loop
    scene.reset()
    target.set_pos(torch.tensor([[3.0, 0.0, 0.1525]] * n_envs, device=device))
    out = cv2.VideoWriter('picar_vision_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, (640, 480))
    
    print("Generating demo video...")
    for i in range(max_steps):
        rgb, _, _, _ = robot_cam.render()
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # Robust demo observation handling
        obs = torch.from_numpy(rgb).float().to(device)
        if obs.dim() == 3:
             # If single image (H,W,C), expand to match envs
             obs = obs.unsqueeze(0).expand(n_envs, -1, -1, -1)

        # Slice just the first env for demo policy inference (though we simulate all)
        obs_demo = obs[0:1] 
        
        mean, _ = policy(obs_demo)
        # Broadcast action to all for simplicity in demo or just control env 0
        actions = mean.repeat(n_envs, 1)
        
        full_cmds.fill_(0)
        full_cmds[:, motor_indices[0]] = actions[:, 0]
        full_cmds[:, motor_indices[1]] = actions[:, 1]
        picar.control_dofs_velocity(full_cmds)
        
        scene.step()
        
        # Render viewer camera following the FIRST car
        car_pos = picar.get_pos()[0].cpu().numpy()
        viewer_cam.set_pose(pos=(car_pos[0]-4, car_pos[1]-4, 3.0), lookat=(car_pos[0], car_pos[1], 0.0))
        
        frame_rgb, _, _, _ = viewer_cam.render()
        
        # Handle viewer frame shape issues
        if frame_rgb.ndim == 4:
            frame_rgb = frame_rgb[0]
        if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3:
             pass # Standard HWC
        
        frame = (frame_rgb * 255).astype(np.uint8)
        # Explicit copy to ensure contiguous array for OpenCV
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print("Demo video saved")

if __name__ == "__main__":
    train()