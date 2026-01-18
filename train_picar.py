import genesis as gs
import torch
import numpy as np
import time

# 1. SETUP
gs.init(backend=gs.gpu)

# 2. SCENE (HEADLESS MODE)
# show_viewer=False prevents the "libEGL" and "dri2" crashes on servers
scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
    ),
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

# 4. BUILD (1000 Parallel Environments)
n_envs = 1000
scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

# 5. GET MOTORS
# We find the joints and get their LOCAL index for the solver
j_left = car.get_joint('rl_wheel_joint')
j_right = car.get_joint('rr_wheel_joint')
rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]

# 6. SIMULATION LOOP
print(f"Starting simulation of {n_envs} cars on GPU...")

# Create velocity tensors
# Shape: (n_envs, 2 motors)
# We use a tensor on the GPU to control all 1000 cars at once
forward_velocity = torch.full((n_envs, 2), 15.0, device='cuda') # 15 rad/s
stop_velocity    = torch.zeros((n_envs, 2), device='cuda')      # 0 rad/s

total_steps = 500 # 5 seconds

start_time = time.time()

for step in range(total_steps):
    
    # CONTROL
    # in Genesis, calling 'control_dofs_velocity' activates the internal velocity controller.
    # You do NOT need to set a "control mode" beforehand.
    if step < 400:
        car.control_dofs_velocity(forward_velocity, rear_wheels_idx)
    else:
        car.control_dofs_velocity(stop_velocity, rear_wheels_idx)

    scene.step()
    
    if step % 100 == 0:
        print(f"Step {step}/{total_steps} completed.")

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")