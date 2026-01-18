import genesis as gs
import torch
import cv2
import time

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
        renderer=gs.renderers.Rasterizer(), # Required for camera sensors
    )

    plane = scene.add_entity(gs.morphs.Plane())
    car = scene.add_entity(gs.morphs.URDF(file='picarx.urdf', fixed=False, pos=(0, 0, 0.1)))

    # 2. Attach Ego-Centric Camera
    # res=(96, 96) limits vision to specific request
    # attached=car locks the camera to the robot chassis
    # pos/lookat become relative offsets: (0.1, 0, 0.1) is roughly "on the hood"
    cam = scene.add_camera(
        res=(96, 96),
        attached=car,
        pos=(0.1, 0.0, 0.15), 
        lookat=(1.0, 0.0, 0.0), # Looking forward relative to car
        fov=60,
        GUI=False
    )

    # Build with 1000 Environments
    n_envs = 1000
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    # Motor Setup
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    forward_velocity = torch.full((n_envs, 2), 15.0, device='cuda')
    stop_velocity    = torch.zeros((n_envs, 2), device='cuda')

    print("Starting Training Loop with Vision...")
    
    # Run Simulation Loop
    for step in range(500):
        # Actuation
        if step < 400:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        
        scene.step()

        # 3. Observation (The Robot "Seeing")
        # This replaces God-Mode state access.
        # rgb shape will be: [1000, 96, 96, 3]
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

    # Camera Setup (Identical to Phase 1)
    # We record exactly what the robot sees (96x96)
    cam = scene.add_camera(
        res=(96, 96),
        attached=car,
        pos=(0.1, 0.0, 0.15),
        lookat=(1.0, 0.0, 0.0), 
        fov=60,
        GUI=False
    )

    scene.build(n_envs=1)

    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    # Video Writer - Note the resolution must match the camera (96x96)
    out = cv2.VideoWriter('last_simulation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (96, 96))

    forward_velocity = torch.tensor([[15.0, 15.0]], device='cuda')
    stop_velocity    = torch.tensor([[0.0, 0.0]], device='cuda')

    for step in range(500):
        if step < 400:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        
        scene.step()

        # Update Camera
        # NO MANUAL UPDATE NEEDED. 
        # The camera is rigidly attached to the car and moves with the physics engine.

        # Record
        rgb, _, _, _ = cam.render(rgb=True)
        if rgb is not None:
            # rgb[0] extracts the single environment image
            image = rgb[0].cpu().numpy()
            # Convert RGB (Genesis) to BGR (OpenCV)
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if step % 50 == 0:
            print(f"Recording frame {step}/500")

    out.release()
    print("✅ Video saved to 'last_simulation.mp4' (96x96 resolution)")

if __name__ == "__main__":
    run_phase_1_training()
    run_phase_2_recording()