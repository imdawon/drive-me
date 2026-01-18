import genesis as gs
import numpy as np
import imageio

# 1. SETUP
gs.init(backend=gs.gpu)

# 2. SCENE
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 720), # HD Resolution for the video
        camera_pos=(1.0, -1.0, 1.0),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=45,
    ),
    show_viewer=False 
)

# 3. ENTITIES
plane = scene.add_entity(gs.morphs.Plane())
car = scene.add_entity(
    gs.morphs.URDF(
        file='picar.urdf',
        pos=(0, 0, 0.1),
        fixed=False
    )
)

# 4. CAMERA FOR RECORDING
# We add a specific camera for rendering the video output
cam = scene.add_camera(
    res=(1280, 720),
    pos=(1.5, -1.5, 1.5),
    lookat=(0, 0, 0),
    fov=45,
    GUI=False
)

scene.build()

# 5. CONTROL SETUP
joints = car.dofs_info_from_names(['rl_wheel_joint', 'rr_wheel_joint'])
rear_wheels_idx = [joints['rl_wheel_joint'], joints['rr_wheel_joint']]
car.set_dofs_control_mode(rear_wheels_idx, gs.MOTION_MODE.VELOCITY)

# 6. SIMULATION & RECORDING
frames = []
total_steps = 500  # 5 Seconds

print("Simulating and recording...")

for step in range(total_steps):
    # --- CONTROL ---
    # Simple logic: Drive forward for 4s, stop for 1s
    if step < 400:
        car.set_dofs_velocity([15.0, 15.0], rear_wheels_idx)
    else:
        car.set_dofs_velocity([0.0, 0.0], rear_wheels_idx)
    
    # --- STEP PHYSICS ---
    scene.step()
    
    # --- UPDATE CAMERA (OPTIONAL: FOLLOW CAR) ---
    # Update camera lookat to follow the car's position
    car_pos = car.get_pos()
    cam.set_pose(
        pos=(car_pos[0] + 1.2, car_pos[1] - 1.2, 1.0), # Keep camera offset relative to car
        lookat=car_pos
    )
    
    # --- RENDER FRAME ---
    # Capture the frame from the camera
    rgb = cam.render()
    frames.append(rgb)

# 7. SAVE VIDEO
print("Saving video to picar_simulation.mp4...")
imageio.mimsave('picar_simulation.mp4', np.stack(frames), fps=60)
print("Done!")