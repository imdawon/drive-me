import genesis as gs
import torch
import cv2
import numpy as np
import time

# 1. SETUP
gs.init(backend=gs.gpu)

# 2. SCENE
# We enable the Rasterizer here so we CAN record video later, 
# even though show_viewer is False.
scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.01),
    rigid_options=gs.options.RigidOptions(enable_collision=True),
    renderer=gs.renderers.Rasterizer(), # <--- REQUIRED FOR VIDEO SAVING
)

# 3. ENTITIES
plane = scene.add_entity(gs.morphs.Plane())

car = scene.add_entity(
    gs.morphs.URDF(
        file='picarx.urdf',
        fixed=False,
        pos=(0, 0, 0.1),
    )
)

# 4. CAMERA
# Position the camera to look at the first car (Env 0)
cam = scene.add_camera(
    res=(640, 480),
    pos=(1.5, -1.5, 1.2),
    lookat=(0, 0, 0.2),
    fov=45,
    GUI=False
)

# 5. BUILD (1000 Parallel Environments)
n_envs = 1000
scene.build(n_envs=n_envs, env_spacing=(1.5, 1.5))

# 6. MOTORS
j_left = car.get_joint('rl_wheel_joint')
j_right = car.get_joint('rr_wheel_joint')
rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

# ==========================================================
# PHASE 1: MASSIVE PARALLEL TRAINING (No Video)
# ==========================================================
print(f"🚀 PHASE 1: Simulating {n_envs} cars (High Speed)...")

forward_velocity = torch.full((n_envs, 2), 15.0, device='cuda')
stop_velocity    = torch.zeros((n_envs, 2), device='cuda')

# Run for 5 seconds
for step in range(500):
    if step < 400:
        car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
    else:
        car.control_dofs_velocity(stop_velocity, rear_wheels_idx)
    scene.step()

print("Phase 1 Complete.")

# ==========================================================
# PHASE 2: THE "LAST SIMULATION" (Video Recording)
# ==========================================================
print("🎥 PHASE 2: Recording Final Video (Environment 0)...")

# 1. Reset everything to start position
scene.reset()

# 2. Setup Video Writer
out = cv2.VideoWriter(
    'last_simulation.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    60, 
    (640, 480)
)

# 3. Run ONE episode just for the camera
for step in range(500):
    
    # We still control all cars, but we only watch Car 0
    if step < 400:
        car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
    else:
        car.control_dofs_velocity(stop_velocity, rear_wheels_idx)

    scene.step()
    
    # Update camera to follow Car 0
    # We grab the position of the FIRST car in the array
    car_pos = car.get_pos()[0].cpu().numpy() 
    
    cam.set_pose(
        pos=(car_pos[0] + 1.2, car_pos[1] - 1.2, 0.8),
        lookat=(car_pos[0], car_pos[1], 0.2)
    )
    
    # Capture Frame
    rgb, _, _, _ = cam.render(rgb=True)
    if rgb is not None:
        # Genesis gives RGB, OpenCV needs BGR
        out.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
    if step % 100 == 0:
        print(f"Recording frame {step}/500")

out.release()
print("✅ Video saved to 'last_simulation.mp4'")