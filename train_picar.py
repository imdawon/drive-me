import genesis as gs
import torch
import cv2
import time
import numpy as np # Required for transform matrices

# 1. SETUP
gs.init(backend=gs.gpu)

def run_phase_1_training():
    print("\n" + "="*50)
    print("🚀 PHASE 1: Massive Parallel Simulation (1000 Cars)")
    print("="*50)

    # 1. Enable Rasterizer for Vision
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        renderer=gs.renderers.Rasterizer(), 
    )

    plane = scene.add_entity(gs.morphs.Plane())
    car = scene.add_entity(gs.morphs.URDF(file='picarx.urdf', fixed=False, pos=(0, 0, 0.1)))

    # 2. Add Camera (Without 'attached' argument)
    cam = scene.add_camera(
        res=(96, 96),
        fov=60,
        GUI=False,
    )

    # 3. Build Scene FIRST
    n_envs = 1000
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    # 4. Attach Camera (The FIX)
    # Mount: 0.1m forward, 0.15m up from car center
    T_cam = np.eye(4)
    T_cam[:3, 3] = np.array([0.1, 0.0, 0.15]) 
    
    # Ensure 'base_link' exists in your URDF, or use car.links[0]
    cam.attach(
        rigid_link=car.get_link('base_link'), 
        offset_T=T_cam
    )

    # Motor Setup
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    forward_velocity = torch.full((n_envs, 2), 15.0, device='cuda')
    stop_velocity    = torch.zeros((n_envs, 2), device='cuda')

    print("Starting Training Loop with Vision...")
    
    # Run Simulation Loop
    for step in range(500):
        if step < 400:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        
        scene.step()

        if step % 100 == 0:
            # Note: The rasterizer returns a NumPy array here
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

    # Camera setup
    cam = scene.add_camera(res=(96, 96), fov=60, GUI=False)

    scene.build(n_envs=1)

    # Attach Camera
    T_cam = np.eye(4)
    T_cam[:3, 3] = np.array([0.1, 0.0, 0.15]) 
    cam.attach(rigid_link=car.get_link('base_link'), offset_T=T_cam)

    # Motors
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    # FIX 1: Use .avi and MJPG (Safer for headless servers)
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

        # Render
        rgb, _, _, _ = cam.render(rgb=True)
        
        if rgb is not None:
            # FIX 2: Handle Data Types and Channels carefully
            image = rgb[0] # Get the first environment
            
            # Debug Print (Only once)
            if step == 0:
                print(f"DEBUG: Raw Genesis Output Shape: {image.shape}, Dtype: {image.dtype}")

            # Ensure valid range 0-255 and uint8 type
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)

            # FIX 3: Handle Channel Count (RGB vs RGBA)
            # Genesis sometimes returns 4 channels (RGBA). OpenCV Writer needs 3 (BGR).
            if image.shape[2] == 4:
                frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Ensure contiguous memory (Crucial for OpenCV writers)
            frame_bgr = np.ascontiguousarray(frame_bgr)
            
            out.write(frame_bgr)
        
        if step % 50 == 0:
            print(f"Recording frame {step}/500")

    out.release()
    print(f"✅ Video saved to '{video_path}'")
    
if __name__ == "__main__":
    run_phase_1_training()
    run_phase_2_recording()