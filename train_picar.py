import genesis as gs
import torch
import cv2
import time

# 1. SETUP
# We initialize the GPU backend once at the very start
gs.init(backend=gs.gpu)

def run_phase_1_training():
    print("\n" + "="*50)
    print("🚀 PHASE 1: Massive Parallel Simulation (1000 Cars)")
    print("="*50)

    # Create the "Training" Scene
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
    )

    plane = scene.add_entity(gs.morphs.Plane())
    car = scene.add_entity(gs.morphs.URDF(file='picarx.urdf', fixed=False, pos=(0, 0, 0.1)))

    # Build with 1000 Environments
    n_envs = 1000
    scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

    # Motor Setup
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    # Command Tensors
    forward_velocity = torch.full((n_envs, 2), 15.0, device='cuda')
    stop_velocity    = torch.zeros((n_envs, 2), device='cuda')

    # Run Simulation Loop
    for step in range(500):
        if step < 400:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        scene.step()
    
    print("Phase 1 Complete. Destroying scene to free resources...")
    # NOTE: In Python, letting 'scene' go out of scope helps, 
    # but strictly speaking, Genesis keeps some data on GPU. 
    # However, building a new scene next usually works fine.
    return

def run_phase_2_recording():
    print("\n" + "="*50)
    print("🎥 PHASE 2: Recording Final Video (1 Car)")
    print("="*50)

    # Create a NEW Scene specifically for recording
    # We add the Rasterizer here to enable the camera
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        renderer=gs.renderers.Rasterizer(), 
    )

    plane = scene.add_entity(gs.morphs.Plane())
    car = scene.add_entity(gs.morphs.URDF(file='picarx.urdf', fixed=False, pos=(0, 0, 0.1)))

    # Camera Setup
    cam = scene.add_camera(
        res=(640, 480),
        pos=(1.5, -1.5, 1.2),
        lookat=(0, 0, 0.2),
        fov=45,
        GUI=False
    )

    # BUILD WITH ONLY 1 ENVIRONMENT
    # This ensures we aren't simulating 999 invisible cars
    scene.build(n_envs=1)

    # Motor Setup (Must re-fetch for new scene)
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

    # Video Writer
    out = cv2.VideoWriter('last_simulation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (640, 480))

    # Tensors for just 1 car
    forward_velocity = torch.tensor([[15.0, 15.0]], device='cuda')
    stop_velocity    = torch.tensor([[0.0, 0.0]], device='cuda')

    for step in range(500):
        # Control
        if step < 400:
            car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
        else:
            car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
        
        scene.step()

        # Update Camera
        car_pos = car.get_pos()[0].cpu().numpy()
        cam.set_pose(
            pos=(car_pos[0] + 1.2, car_pos[1] - 1.2, 0.8),
            lookat=(car_pos[0], car_pos[1], 0.2)
        )

        # Record
        rgb, _, _, _ = cam.render(rgb=True)
        if rgb is not None:
            out.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        if step % 50 == 0:
            print(f"Recording frame {step}/500")

    out.release()
    print("✅ Video saved to 'last_simulation.mp4'")

if __name__ == "__main__":
    # Execute the two phases sequentially
    run_phase_1_training()
    run_phase_2_recording()