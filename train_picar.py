import genesis as gs
import torch
import cv2
import time
import numpy as np

# 1. SETUP
gs.init(backend=gs.gpu)

def get_camera_transform():
    # 1. Start with Identity
    T = np.eye(4)
    
    # 2. Position (Forward 0.1, Up 0.15)
    T[:3, 3] = np.array([0.1, 0.0, 0.15])
    
    # 3. Rotation (The Fix)
    # Camera default: Looks down -Z axis, Top is +Y axis
    
    # Step A: Rotate around X by +90 degrees 
    # (Lifts camera from looking DOWN to looking HORIZON)
    rot_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    
    # Step B: Rotate around Z by -90 degrees 
    # (Turns camera to the RIGHT)
    # Note: If it faces Left, change to +90 (standard rotation matrices can vary by handedness)
    rot_z = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    
    # Combine: Apply X then Z
    rot_combined = rot_z @ rot_x
    
    T[:3, :3] = rot_combined
    return T

def run_simulation(is_recording=False):
    print("\n" + "="*50)
    mode_name = "RECORDING" if is_recording else "TRAINING"
    print(f"🚀 PHASE: {mode_name}")
    print("="*50)

    # --- FIX 1: PHYSICS STABILITY ---
    # dt=0.01 with substeps=10 means physics runs at 1000Hz (1ms steps)
    # This solves the "wobble/jitter" on the ground.
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=0.01, 
            substeps=10  
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        renderer=gs.renderers.Rasterizer(), 
    )

    plane = scene.add_entity(gs.morphs.Plane())
    
    # Load car
    car = scene.add_entity(
        gs.morphs.URDF(
            file='picarx.urdf', 
            fixed=False, 
            pos=(0, 0, 0.1)
        )
    )

    # Setup Camera
    cam = scene.add_camera(res=(96, 96), fov=60, GUI=False)

    # Build Scene (Required before attaching)
    n_envs = 1 if is_recording else 1000
    scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))

    # --- FIX 2: CAMERA ATTACHMENT ---
    # We attach using the corrected transform matrix
    cam.attach(
        rigid_link=car.get_link('base_link'), 
        offset_T=get_camera_transform()
    )

    # --- FIX 3: MOTOR CONTROL MODES ---
    # We must set stiffness (kp) to 0 for velocity control to work.
    # Otherwise, the internal P-controller tries to hold the wheel at position 0.
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]
    
    # Set KP (stiffness) to 0, KV (damping) to something small (e.g., 0.5 or 1.0)
    # Note: Genesis requires setting this for all DOFs or specific ones.
    # We iterate over the wheel indices to set their gains.
    dofs = car.n_dofs
    # Create gain arrays
    kps = np.array([0.0] * dofs)     # Zero stiffness (allows free spinning)
    kvs = np.array([10.0] * dofs)    # Some damping for stability
    
    # Apply gains to the car
    car.set_dofs_kp(kps)
    car.set_dofs_kv(kvs)

    # Prepare Tensors
    forward_velocity = torch.full((n_envs, 2), 25.0, device='cuda') # Increased speed slightly
    stop_velocity    = torch.zeros((n_envs, 2), device='cuda')

    # Video Writer Setup (Only if recording)
    out = None
    if is_recording:
        video_path = 'last_simulation.avi'
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_path, fourcc, 30, (96, 96))
        print("🎥 Video Writer Initialized")

    print("Starting Loop...")
    
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
                # Handle Shape: (H, W, C) or (1, H, W, C)
                if rgb.ndim == 4: image = rgb[0]
                else: image = rgb
                
                # Normalize and Color Convert
                if image.dtype != np.uint8:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                
                # Check channel count (RGB vs RGBA)
                if image.shape[2] == 4:
                    frame = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                else:
                    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                out.write(frame)
        
        if step % 50 == 0:
             print(f"Step {step}: Running...")

    if out:
        out.release()
        print(f"✅ Video saved to {video_path}")

    print("Done.")

if __name__ == "__main__":
    # Run Phase 1 (Training Check)
    run_simulation(is_recording=False)
    
    # Run Phase 2 (Video Recording)
    run_simulation(is_recording=True)