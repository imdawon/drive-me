import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import numpy as np
import cv2

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
    urdf_content = """[paste the URDF xml above here]"""
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
    # Viewer camera for demo video
    viewer_cam = scene.add_camera(res=(640, 480), pos=(8.0, -8.0, 5.0), lookat=(0.0, 0.0, 0.0), fov=60)
    # Robot-mounted camera for observations (96x96 egocentric, attached to camera_link looking forward)
    robot_cam = scene.add_camera(res=(96, 96), fov=70, near=0.05, far=10.0)
    # Assuming Genesis supports vectorized egocentric rendering from robot-mounted camera
    scene.build(n_envs=2048)
    return scene, picar, target, viewer_cam, robot_cam

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
        self.act_scale = 21.0  # ~200 RPM at 6V with ~35mm radius

    def forward(self, x):
        # x: (b, 96, 96, 3) float 0-1
        x = x.permute(0, 3, 1, 2)  # to (b, 3, 96, 96)
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

    full_cmds = torch.zeros((2048, 12), device="cpu")  # adjust if needed

    max_steps = 500
    contact_threshold = 0.45

    for epoch in range(30):
        scene.reset()

        # Randomize robot orientation (position fixed)
        robot_angles = torch.rand(2048, device="cpu") * 2 * np.pi
        robot_quat = gs.math.euler_to_quat(torch.stack([torch.zeros(2048), torch.zeros(2048), robot_angles], dim=1).to("cpu"))
        picar.set_quat(robot_quat)

        # Randomize target position (0.6-3.0m distance)
        target_angles = torch.rand(2048, device="cpu") * 2 * np.pi
        target_dists = 0.6 + torch.rand(2048, device="cpu") * 2.4
        target_x = target_dists * torch.cos(target_angles)
        target_y = target_dists * torch.sin(target_angles)
        target_pos = torch.stack([target_x, target_y, torch.full((2048,), 0.1525)], dim=1)
        target.set_pos(target_pos)

        log_probs_list = []
        entropies_list = []
        rewards_list = []

        done = torch.zeros(2048, dtype=torch.bool, device="cpu")

        for step in range(max_steps):
            # Egocentric RGB observations (assuming vectorized render from attached robot_cam)
            rgb = robot_cam.render()  # (2048, 96, 96, 3) float or uint8
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

            pos = picar.get_pos()
            rel_xy = pos[:, :2] - target_pos[:, :2]
            old_dist = torch.norm(rel_xy, dim=1)
            new_dist = torch.norm(pos[:, :2] - target_pos[:, :2], dim=1)

            quat = picar.get_quat()
            yaw = torch.atan2(2 * (quat[:,0] * quat[:,3] + quat[:,1] * quat[:,2]),
                              1 - 2 * (quat[:,2]**2 + quat[:,3]**2))
            desired_yaw = torch.atan2(rel_xy[:,1], rel_xy[:,0])
            diff = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
            heading_bonus = torch.cos(diff)

            reward = (old_dist - new_dist) * 10.0
            reward += heading_bonus
            reward -= (action_cpu ** 2).mean(dim=1) * 0.0001
            reward -= 0.02  # time penalty

            contact = new_dist < contact_threshold
            new_contact = contact & ~done
            reward[new_contact] += 200.0

            done = done | contact
            reward[done & (torch.arange(step+1, device="cpu") > 0)] = 0.0  # optional: zero after done

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

        final_dist = new_dist.mean().item()
        success_rate = contact.float().mean().item()
        print(f"Epoch {epoch}: Avg final dist {final_dist:.2f}, Success {success_rate:.3f}")

    torch.save(policy.state_dict(), "picar_vision_policy.pth")

    # Demo video on first env
    scene.reset()
    target.set_pos(torch.tensor([[3.0, 0.0, 0.1525]] * 2048))
    out = cv2.VideoWriter('picar_vision_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, (640, 480))

    for i in range(max_steps):
        rgb = robot_cam.render()
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        obs_gpu = rgb.to(device)

        mean, _ = policy(obs_gpu)
        actions_cpu = mean.cpu()
        actions_cpu[done] = 0.0

        full_cmds.fill_(0)
        full_cmds[:, motor_indices[0]] = actions_cpu[:, 0]
        full_cmds[:, motor_indices[1]] = actions_cpu[:, 1]

        picar.control_dofs_velocity(full_cmds)
        scene.step()

        car_pos = picar.get_pos()[0].cpu().numpy()
        viewer_cam.set_pose(pos=(car_pos[0]-3, car_pos[1]-3, 3.0), lookat=car_pos)

        frame_rgb, _, _, _ = viewer_cam.render()
        frame_rgb = frame_rgb.cpu().numpy()[0] if frame_rgb.ndim == 4 else frame_rgb.cpu().numpy()
        frame = (frame_rgb * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print("Demo video saved")

if __name__ == "__main__":
    train()