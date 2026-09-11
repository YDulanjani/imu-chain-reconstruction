# imu-chain-reconstruction
Generating synthetic IMU chain data and reconstructing shapes from quaternion measurements.


random_walk_logic.py 
Generates synthetic chains
  ground-truth positions 
  quaternion values
  Optional noise on the quaternions


chain_reconstruction_evaluator_logic.py
Reconstruct X, Y, Z positions from quaternions by direct mathematics
  Measure the error against ground truth.
