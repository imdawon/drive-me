import genesis as gs
import torch
import cv2
import time
import numpy as np 

# 1. SETUP
gs.init(backend=gs.gpu)

def get_camera_transform():
    """
    Creates a transform matrix that offsets the camera and 
    rotates it 90 degrees up (Y-axis) and 90 degrees right (Z-axis).
    """
    T_cam = np.eye(4)
    
    # 1. Position: 0.1m forward, 0.15m up
    T_cam[:3, 3] = np.array([0.1, 0.0, 0.15])
    
    # 2. Rotation Matrices
    # Rotate 90 degrees Up (Pitch around Y-axis)
    # This turns a camera looking down (-Z) to look forward (+X)
    theta_y = np.radians(90)
    rot_y = np.array([
        [np.cos(theta_y),  0, np.sin(theta_y)],
        [0,                1, 0              ],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # Rotate 90 degrees Right (Yaw around Z-axis)
    # Note: Depending on coordinate system, -90 might be "right". 
    # I used -90 here as it's standard for "Right" in right-handed systems.
    # If it faces Left, change -90 to 90.
    theta_z = np.radians(0) 
    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0,                0,               1]
    ])

    # Combine Rotations: Apply Y first (lift head), then Z (turn head)
    # Matrix multiplication order: R_combined = R_z @ R_y
    rot_combined = rot_z @ rot_y
    
    # Apply to T_cam
    T_cam[:3, :3] = rot_combined
    return T_cam

def run_phase_1_training():
    print("\n" + "="*50)
    print("🚀 PHASE 1: Massive Parallel Simulation (1000 Cars)")
    print("="*50)

    scene = gs.Scene(
        show_viewer=False,
        # Update 1: Increase simulation frequency or substeps
        sim_options=gs.options.SimOptions(
            dt=0.01,       # Control frequency (100Hz)
            substeps=4,    # Physics frequency (400Hz) - SMOOTHS CONTACTS
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
        ),
        renderer=gs.renderers.Rasterizer(), 
    )

    plane = scene.add_entity(gs.morphs.Plane())
    car = scene.add_entity(gs.morphs.URDF(file='picarx.urdf', fixed=False, pos=(0, 0, 0.1)))

    cam = scene.add_camera(res=(96, 96), fov=60, GUI=False)

    n_envs = 1000
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    # --- FIX APPLIED HERE ---
    T_cam = get_camera_transform()
    
    cam.attach(
        rigid_link=car.get_link('base_link'), 
        offset_T=T_cam
    )

    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    forward_velocity = torch.full((n_envs, 2), 15.0, device='cuda')
    stop_velocity    = torch.zeros((n_envs, 2), device='cuda')

    print("Starting Training Loop with Vision...")
    
    for step in range(500):
        if step < 400:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        
        scene.step()

        if step % 100 == 0:
            rgb, _, _, _ = cam.render(rgb=True)
            print(f"Step {step}: Generated Observation Tensor {rgb.shape}")
    
    print("Phase 1 Complete.")
    return

def run_phase_2_recording():
    print("\n" + "="*50)
    print("🎥 PHASE 2: Recording Final Video (Ego-View)")
    print("="*50)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        renderer=gs.renderers.Rasterizer(), 
    )

    plane = scene.add_entity(gs.morphs.Plane())
    car = scene.add_entity(gs.morphs.URDF(file='picarx.urdf', fixed=False, pos=(0, 0, 0.1)))

    cam = scene.add_camera(res=(96, 96), fov=60, GUI=False)

    scene.build(n_envs=1)

    # --- FIX APPLIED HERE ---
    T_cam = get_camera_transform()
    
    cam.attach(rigid_link=car.get_link('base_link'), offset_T=T_cam)

    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    video_path = 'last_simulation.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, 60, (96, 96))

    forward_velocity = torch.tensor([[15.0, 15.0]], device='cuda')
    stop_velocity    = torch.tensor([[0.0, 0.0]], device='cuda')

    print("Starting recording...")

    for step in range(500):
        if step < 400:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        
        scene.step()

        rgb, _, _, _ = cam.render(rgb=True)
        
        if rgb is not None:
            if rgb.ndim == 4:
                image = rgb[0]
            else:
                image = rgb

            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)

            if image.shape[2] == 4:
                frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_bgr = np.ascontiguousarray(frame_bgr)
            out.write(frame_bgr)
        
        if step % 50 == 0:
            print(f"Recording frame {step}/500")

    out.release()
    print(f"✅ Video saved to '{video_path}'")
    
if __name__ == "__main__":
    run_phase_1_training()
    run_phase_2_recording()