import genesis as gs
import torch
import time

# 1. SETUP
gs.init(backend=gs.gpu)

# 2. SCENE (HEADLESS MODE ENABLED)
# show_viewer=False prevents the EGL/Rendering crash on servers
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

# 5. GET MOTORS (THE FIX)
# We use .get_joint() and .dof_idx_local to avoid the AttributeError
try:
    j_left = car.get_joint('rl_wheel_joint')
    j_right = car.get_joint('rr_wheel_joint')
    
    # Store the indices for the solver
    rear_wheels_idx = [j_left.dof_idx_local, j_right.dof_idx_local]
    
except AttributeError as e:
    print(f"CRITICAL ERROR: Could not find joints. Check URDF joint names. {e}")
    exit(1)

# Set control mode to VELOCITY for the rear wheels
car.set_dofs_control_mode(rear_wheels_idx, gs.MOTION_MODE.VELOCITY)

# 6. SIMULATION LOOP
print(f"Starting simulation of {n_envs} cars on RTX 3090...")

# Create tensors for forward and stop
# Shape: (n_envs, 2 motors)
forward_velocity = torch.full((n_envs, 2), 15.0, device='cuda') # 15 rad/s
stop_velocity    = torch.zeros((n_envs, 2), device='cuda')      # 0 rad/s

total_steps = 500 # 5 seconds

start_time = time.time()

for step in range(total_steps):
    
    # Drive for 4 seconds (400 steps), then Stop
    if step < 400:
        car.set_dofs_velocity(forward_velocity, rear_wheels_idx)
    else:
        car.set_dofs_velocity(stop_velocity, rear_wheels_idx)

    scene.step()
    
    if step % 100 == 0:
        print(f"Step {step}/{total_steps} completed.")

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")