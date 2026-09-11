# Pragmatic full-network hover prototype

This is a compact record of a deliberately behavior-first experiment. Starting from the
stopped update-50 assisted-throttle controller, a 20-generation antithetic evolution
strategy tuned 24 shared quantities at the existing eight front-leg motor pools. The
selected values were folded into native neuron biases and signed incoming-edge magnitudes;
there is no adapter, extra recurrent model, or controller outside the MaleCNS graph.

On 128 new airborne six-second cases spanning unseen absolute marker heights and unseen
marker/camera-style combinations, the all-native actor achieved 50 strict height holds,
13.8 cm mean height RMSE, 0.080 m/s vertical-speed RMS, 3.60-degree tilt RMS, and no ground
contacts or invalid flights. The untouched full-network source achieved no strict holds
and 64.9 cm mean height RMSE on the same cases. Freezing each episode's visual input at the
marker-change time reduced the selected controller to 8 strict holds, raised height RMSE
to 37.5 cm, and produced six ground/invalid cases.

The actor receives 320x200 RGB at 125-degree horizontal FOV plus roll and pitch. Its only
history is the recurrent state of all 165,122 MaleCNS neurons, and its roll, pitch, yaw and
throttle outputs all pass through the two front legs and virtual sticks. No accelerometer,
height, velocity, mass, hover-thrust, previous-action, phase, or teacher action enters the
actor.

This is useful proof-of-concept hover, not a promoted milestone: the aircraft starts
airborne with settled sticks, only 39.1% of cases meet the strict 10 cm per-episode height
criterion, and the checkpoint has not yet been tested for takeoff or gate flight. The
105 MB exploratory checkpoint remains under ignored `runs/` storage. It should use Git
LFS if a later result is promoted into the repository.

