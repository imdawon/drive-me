import genesis as gs
import genesis.utils.geom as gu
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
import cv2

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
gs.init(backend=gs.gpu if device == "cuda" else gs.cpu, logging_level="warning")

# --- Helper ---
def euler_to_quat(euler):
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ], dim=-1)

# --- URDF ---
def create_urdf():
    urdf = """<?xml version="1.0"?><robot name="picar"><link name="base_link">
    <visual><geometry><box size="0.26 0.17 0.06"/></geometry><material name="g"><color rgba="0.7 0.7 0.7 1"/></material></visual>
    <collision><geometry><box size="0.26 0.17 0.06"/></geometry></collision>
    <inertial><mass value="1.8"/><inertia ixx="0.08" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.08"/></inertial></link>
    <link name="cam"><visual><geometry><box size="0.04 0.06 0.04"/></geometry><material name="b"><color rgba="0.2 0.2 0.2 1"/></material></visual>
    <collision><geometry><box size="0.04 0.06 0.04"/></geometry></collision>
    <inertial><mass value="0.01"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
    <joint name="cam_joint" type="fixed"><parent link="base_link"/><child link="cam"/><origin xyz="0.14 0 0.10" rpy="0 -0.17 0"/></joint>
    """
    for x, y, n in [(0.13, 0.085, "fl"), (0.13, -0.085, "fr"), (-0.13, 0.085, "rl"), (-0.13, -0.085, "rr")]:
        urdf += f"""<link name="w_{n}"><visual><geometry><cylinder length="0.03" radius="0.032"/></geometry><origin rpy="1.57 0 0"/><material name="w"><color rgba="1 1 1 1"/></material></visual>
        <collision><geometry><cylinder length="0.03" radius="0.032"/></geometry><origin rpy="1.57 0 0"/></collision>
        <inertial><mass value="0.05"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
        <joint name="j_{n}" type="continuous"><parent link="base_link"/><child link="w_{n}"/><origin xyz="{x} {y} -0.032"/><axis xyz="0 1 0"/><limit effort="20" velocity="40"/></joint>"""
    urdf += "</robot>"
    with open("picar.urdf", "w") as f: f.write(urdf)
    return "picar.urdf"

# --- Scene ---
def create_scene():
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.02, substeps=4), show_viewer=False)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.2))
    picar = scene.add_entity(gs.morphs.URDF(file=create_urdf(), pos=(0, 0, 0.035), fixed=False))
    target = scene.add_entity(gs.morphs.Box(size=(0.4, 0.25, 0.3), fixed=True), material=gs.materials.Rigid(friction=1.0))
    
    # Cameras
    cam = scene.add_camera(res=(96, 96), pos=(0,0,0), lookat=(1,0,0), fov=70)
    # Fix attachment
    cam.attach(picar.get_link("base_link"), gu.trans_quat_to_T(np.array([0.14, 0.0, 0.10]), np.array([0.996, 0.0, -0.085, 0.0])))
    
    scene.build(n_envs=2048)
    return scene, picar, target, cam

# --- Policy (Fixed Math) ---
class VisionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        # FIXED: 96x96 input -> 64 * 8 * 8 = 4096 features
        self.mean = nn.Linear(4096, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        return torch.tanh(self.mean(self.cnn(x))) * 21.0, torch.exp(self.log_std.clamp(-20, 2))

# --- Training ---
def train():
    scene, picar, target, cam = create_scene()
    policy = VisionPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=3e-4)
    motor_ids = [picar.get_joint(n).dofs_idx_local[0] for n in ["j_rl", "j_rr"]]
    
    for epoch in range(40):
        scene.reset()
        # Randomize
        picar.set_quat(euler_to_quat(torch.stack([torch.zeros(2048, device=device), torch.zeros(2048, device=device), torch.rand(2048, device=device)*6.28], 1)))
        t_pos = torch.stack([(d:=0.6+torch.rand(2048, device=device)*2.4)*torch.cos(a:=torch.rand(2048, device=device)*6.28), d*torch.sin(a), torch.full((2048,), 0.15, device=device)], 1)
        target.set_pos(t_pos)
        
        last_dist = torch.norm(picar.get_pos()[:, :2] - t_pos[:, :2], dim=1)
        done = torch.zeros(2048, dtype=torch.bool, device=device)
        
        for _ in range(500):
            if done.all(): break
            rgb = torch.from_numpy(cam.render()[0]).float().to(device) / 255.0
            
            # Simple shape fix if single frame
            if rgb.ndim == 3: rgb = rgb.unsqueeze(0).expand(2048, -1, -1, -1)
            
            mean, std = policy(rgb)
            dist = D.Normal(mean, std)
            action = dist.rsample()
            
            # Control
            cmds = torch.zeros((2048, 20), device=device)
            cmds[:, motor_ids[0]] = (act := action.clone().masked_fill(done.unsqueeze(1), 0))[:, 0]
            cmds[:, motor_ids[1]] = act[:, 1]
            picar.control_dofs_velocity(cmds)
            scene.step()
            
            # Reward
            curr_dist = torch.norm(picar.get_pos()[:, :2] - t_pos[:, :2], dim=1)
            rew = (last_dist - curr_dist) * 10.0 - 0.02
            
            # PPO Step (Simplified)
            ret = rew # using immediate reward as proxy for simplified example
            loss = -dist.log_prob(action).sum(-1).mean() * ret.mean() 
            opt.zero_grad(); loss.backward(); opt.step()
            
            done |= (curr_dist < 0.45)
            last_dist = curr_dist
            
        print(f"Epoch {epoch}: Success {done.float().mean():.2f}")
    
    torch.save(policy.state_dict(), "policy.pth")

if __name__ == "__main__":
    train()