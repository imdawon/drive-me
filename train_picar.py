import genesis as gs
import torch
import cv2
import numpy as np
import time

# --- PHYSICS CONSTANTS (1 m/s Calibration) ---
# Wheel Radius: 0.0325m | Target: 1.0 m/s
TARGET_SPEED_RADS = 30.8   
TORQUE_LIMIT_NM   = 0.15   
SIM_BOOST         = 1.5    

def get_camera_transform():
    T = np.eye(4)
    # Position: Forward 0.1m, Up 0.15m
    T[:3, 3] = np.array([0.1, 0.0, 0.15])
    # Rotation: Lift 90 deg (X), then Turn 90 deg (Z)
    rot_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rot_z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    T[:3, :3] = rot_z @ rot_x
    return T

def run_simulation(is_recording=False):
    print("\n" + "="*50)
    mode = "RECORDING (1 Car)" if is_recording else "TRAINING (1000 Cars)"
    print(f"🚀 PHASE: {mode}")
    print(f"   Target Speed: {TARGET_SPEED_RADS:.2f} rad/s (~1.0 m/s)")
    print("="*50)

    # 1. SETUP
    gs.init(backend=gs.gpu)

    # 2. SCENE CREATION
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=0.01, 
            substeps=10  
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            gravity=(0, 0, -9.8),
            # REMOVED: default_restitution is not a valid global option
        ),
        renderer=gs.renderers.Rasterizer(), 
    )

    plane = scene.add_entity(gs.morphs.Plane())
    
    # Load Car
    car = scene.add_entity(
        gs.morphs.URDF(file='picarx.urdf', fixed=False, pos=(0, 0, 0.1))
    )

    # Setup Camera
    cam = scene.add_camera(res=(96, 96), fov=60, GUI=False)

    # 3. BUILD
    n_envs = 1 if is_recording else 1000
    scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))

    # 4. ATTACH CAMERA
    cam.attach(
        rigid_link=car.get_link('base_link'), 
        offset_T=get_camera_transform()
    )

    # 5. MOTOR CONTROL & GAINS
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]
    
    dofs = car.n_dofs
    
    # Velocity Control Mode: KP=0, KV=5.0
    kps = np.array([0.0] * dofs)    
    kvs = np.array([5.0] * dofs)    
    
    car.set_dofs_kp(kps)
    car.set_dofs_kv(kvs)
    
    # Torque Limits
    limit_val = TORQUE_LIMIT_NM * SIM_BOOST
    car.set_dofs_force_range(
        lower=np.array([-limit_val] * dofs), 
        upper=np.array([limit_val] * dofs)
    )

    # 6. COMMAND TENSORS
    forward_velocity = torch.full((n_envs, 2), TARGET_SPEED_RADS, device='cuda')
    stop_velocity    = torch.zeros((n_envs, 2), device='cuda')

    # Video Writer
    out = None
    if is_recording:
        video_path = 'last_simulation.avi'
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_path, fourcc, 30, (96, 96))

    print("Starting Loop...")
    
    # 7. SIMULATION LOOP
    for step in range(200):
        # Drive for 150 steps, then stop
        if step < 150:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        
        scene.step()

        if is_recording:
            rgb, _, _, _ = cam.render(rgb=True)
            if rgb is not None:
                if rgb.ndim == 4: image = rgb[0]
                else: image = rgb
                
                if image.dtype != np.uint8:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                
                if image.shape[2] == 4:
                    frame = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                else:
                    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                out.write(frame)
        
        # Periodic Logging
        if step % 50 == 0:
            vel = car.get_v()[0] 
            speed = np.linalg.norm(vel[:2].cpu().numpy())
            print(f"Step {step}: Env 0 Speed = {speed:.3f} m/s")

    if out:
        out.release()
        print(f"✅ Video saved to {video_path}")
    
    return

if __name__ == "__main__":
    run_simulation(is_recording=False)
    run_simulation(is_recording=True)