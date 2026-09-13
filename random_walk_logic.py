# Random-walk IMU chain generator.
# Generates synthetic IMU chain data and quaternion measurements.
# No Streamlit dependency.

#Reference: https://arxiv.org/abs/1704.06053 
#Using Inertial Sensors for Position and Orientation Estimation

import numpy as np
import csv


# Default simulation parameters
NUM_SENSORS = 15
LINK_LENGTH_CM = 5.0
BEND_STD_DEG = 15.0
NUM_SAMPLES = 10

REFERENCE_AXIS = np.array([1.0, 0.0, 0.0])
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
MAX_BEND_SAFETY_RAD = np.pi * 0.92


class ChainSample:
    # Generated chain data for one sample.

    def __init__(self, positions, directions, quats):
        self.positions = positions
        self.directions = directions
        self.quats = quats

    @property
    def num_sensors(self):
        return self.positions.shape[0]


def random_unit_vector(rng):
    # Generate a random unit vector.
    vector = rng.standard_normal(3)
    return vector / np.linalg.norm(vector)


def free_axes_from_fixed(fixed_axes):
    # Convert fixed axis names to their indices.
    # Keep this flexible so x, y, or z can be fixed.
    try:
        fixed_idx = {AXIS_INDEX[axis] for axis in fixed_axes}
    except KeyError as exc:
        raise ValueError(
            f"fixed_axes entries must be 'x', 'y', and/or 'z'; "
            f"got {fixed_axes!r}"
        ) from exc

    free = [i for i in range(3) if i not in fixed_idx]

    if not free:
        raise ValueError(
            "At least one axis must stay free."
        )

    return free


#this it to use in S1 position when we need random values
def random_direction_in_subspace(
    rng,
    free_axes,
):
    # Generate a random direction using the available axes.
    direction = np.zeros(3)

    if len(free_axes) == 1:
        direction[free_axes[0]] = 1.0
        return direction

    if len(free_axes) == 2:
        theta = rng.uniform(0.0, 2 * np.pi)
        i, j = free_axes
        direction[i] = np.cos(theta)
        direction[j] = np.sin(theta)
        return direction

    return random_unit_vector(rng)


def perp_basis(d):
    # Find two unit vectors perpendicular to the current direction.

    #trying to choose the arbitary vector normally choose X
    #but we have an edge case here is if the direction is very close to X then we choose Y as the arbitrary vector
    #becoz if we choose X then the cross product will be zero vector and we cannot get the perpendicular basis
    #(cross product becomes problematic if the two vectors are parallel or nearly parallel)
    arbitrary = (
        np.array([1.0, 0.0, 0.0])
        if abs(d[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )

    u = np.cross(d, arbitrary)
    u = u / np.linalg.norm(u)

    v = np.cross(d, u)

    return u, v


# takes the current chain direction and bends it by a random amount in a random direction.
def bend_direction(
    rng,
    direction,
    bend_std_deg,
):
    # Apply one random bend to the current direction.
    # generates a random number from a normal distribution.
    #taking abs to make angle positive and then converting it to radians
    bend_rad = abs(rng.standard_normal()) * np.radians(bend_std_deg)
    phi = rng.uniform(0.0, 2 * np.pi) #choose which direction to bend in the perpendicular plane

    u, v = perp_basis(direction) #get two directions perpendicular to the current direction
    perp = np.cos(phi) * u + np.sin(phi) * v # get a random direction in the perpendicular plane.

    #Using the Rodrigues' rotation formula to rotate the direction vector around the perpendicular axis by the bend angle
    #v' = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))
    new_direction = (
        np.cos(bend_rad) * direction
        + np.sin(bend_rad) * perp
    )

    return new_direction / np.linalg.norm(new_direction)

#this fuction is to bend the direction in the subspace defined by the free axes
#while keeping the fixed axes at zero. (we need this when 2D contrains needed)
def bend_direction_in_subspace(
    rng,
    direction,
    bend_std_deg,
    free_axes,
):
    
    # If there is only one free axis 
    # we cannot bend the direction.
    if len(free_axes) == 1:
        return direction.copy()

    # If there are two free axes, 
    # we can rotate the direction vector within the plane defined by those axes.
    if len(free_axes) == 2:
        # Rotate within the constrained plane.
        i, j = free_axes

        # Generate a random bend angle within the allowed range
        angle = np.clip(
            rng.standard_normal() * np.radians(bend_std_deg),
            -MAX_BEND_SAFETY_RAD,
            MAX_BEND_SAFETY_RAD,
        )


        c, s = np.cos(angle), np.sin(angle)

        # Rotate the direction vector in the plane defined by the two free axes
        new_direction = direction.copy()
        new_direction[i] = direction[i] * c - direction[j] * s
        new_direction[j] = direction[i] * s + direction[j] * c

        return new_direction / np.linalg.norm(new_direction)

    # If all three axes are free
    # we can use the normal bending function without restrictions
    return bend_direction(rng, direction, bend_std_deg)

# This is to find the quaternion that rotates the reference axis(a) to the new direction vector(b).
#Since we have a fixed reference axis this functions gives us 
#What rotation takes the original +X direction(a) to the new seonsor's direction(b) 
def vec_to_quat(a, b):

    # Calculate the shortest-arc quaternion from a to b.
    #We can use the dot product to find the angle between the two vectors.
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    #1=exactly same direction,0=  90 deg apart, -1 = opposite directions

    # If the vectors are nearly opposite
    # we need to find a perpendicular axis to rotate around
    # becoz if we just take the cross product of two opposite vectors we get a zero vector which is not valid for rotation 
    # (cross product becomes problematic if the two vectors are parallel or nearly parallel)
    if dot < -0.999999:
 
        # find some axis that is perpendicular to a
        axis = np.cross([1.0, 0.0, 0.0], a)

        #If we have a as fixed reference axis (1,0,0) 
        #then the cross product will be zero vector 
        # and we cannot get the perpendicular axis
        #So we can try Y then  this will give us Z axis as the perpendicular axis to rotate around
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross([0.0, 1.0, 0.0], a)

        
        axis = axis / np.linalg.norm(axis)
        #returning the quaternion - 180-degree rotation around the perpendicular axis
        return np.array([0.0, axis[0], axis[1], axis[2]])

    #If vectores aare not opposite we can go on happy path
    cross = np.cross(a, b)
    w = 1.0 + dot

    quat = np.array([w, cross[0], cross[1], cross[2]])

    return quat / np.linalg.norm(quat)


class NoiseSpec:
    # Configuration for one quaternion noise source.

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value


#This is for use in the UI to set the default values for the noise parameters and their ranges.

#Only two noise types available right now and we need to add noise types we want here
# heading noise - rotates the sensor heading by a random angle before converting it to a quaternion
# quat_jitter - adds the noise directly to to quaternion values and renormalizes the result
NOISE_KIND_DEFAULTS = {
    "heading": dict(
        label="Heading angle noise (deg)",
        default=5.0,
        min=0.0,
        max=90.0,
        step=0.5,
        help=(
            "Rotates each sensor heading by a random angle before "
            "converting it to a quaternion."
        ),
    ),
    "quat_jitter": dict(
        label="Quaternion component jitter (sigma)",
        default=0.05,
        min=0.0,
        max=0.5,
        step=0.01,
        help=(
            "Adds Gaussian noise to quaternion components and "
            "renormalizes the result."
        ),
    ),
}



# This function rotates the reference axis by the quaternion value
# this will show us which way the sensor's X-axis is pointing right now
def rotate_ref_by_quat(q):
    # Rotate the reference axis using the quaternion.
    w, x, y, z = q
    qv = np.array([x, y, z])

    t = 2.0 * np.cross(qv, REFERENCE_AXIS)

    return REFERENCE_AXIS + w * t + np.cross(qv, t)


#Here we are applying all the noise specificed in the specs list to the quaternions in order way ()
def apply_quaternion_noise(
    quats,
    specs,
    rng,
):
    noisy_quats = quats.copy()

    for spec in specs:
        if spec.kind == "heading":
            for i in range(len(noisy_quats)):
                direction = rotate_ref_by_quat(noisy_quats[i]) #extracting the current direction
                direction /= np.linalg.norm(direction)

                #randomly bending the direction by the specified angle in the noise spec
                perturbed = bend_direction(
                    rng,
                    direction,
                    spec.value,
                )

                #converting the perturbed direction back to a quaternion
                noisy_quats[i] = vec_to_quat(
                    REFERENCE_AXIS,
                    perturbed,
                )
        
        elif spec.kind == "quat_jitter":

            # Add Gaussian noise to each quaternion component and renormalize
            noisy_quats += rng.normal(
                0.0,
                spec.value,
                size=noisy_quats.shape,
            )

            noisy_quats /= np.linalg.norm(
                noisy_quats,
                axis=1,
                keepdims=True,
            )

        else:
            raise ValueError(f"Unknown noise kind: {spec.kind!r}")

    return noisy_quats


#Generating random walk chain

def generate_chain(
    num_sensors=NUM_SENSORS,
    link_length_cm=LINK_LENGTH_CM,
    bend_std_deg=BEND_STD_DEG,
    randomize_start_orientation=True,
    fixed_axes=(),
    seed=None,
):
    rng = np.random.default_rng(seed)
    free_axes = free_axes_from_fixed(fixed_axes)

    if randomize_start_orientation:
        direction = random_direction_in_subspace(
            rng,
            free_axes,
        )
    elif 0 in free_axes:
        # If the +X axis is free, we can start with it.
        direction = REFERENCE_AXIS.copy()
    else:
        #Not logical
        raise ValueError(
            "Cannot start along the global +X axis while 'x' is fixed. "
            "Enable randomize_start_orientation or leave 'x' free."
        )

    # Initialize the starting position at the origin
    position = np.zeros(3)

    # Initialize arrays to hold the positions, directions, and quaternions for each sensor
    positions = np.zeros((num_sensors, 3))
    directions = np.zeros((num_sensors, 3))
    quats = np.zeros((num_sensors, 4))

    for i in range(num_sensors):
        if i > 0:
            direction = bend_direction_in_subspace(
                rng,
                direction,
                bend_std_deg,
                free_axes,
            )

        # Update the position based on the current direction and link length
        position = position + direction * link_length_cm

        positions[i] = position
        directions[i] = direction
        quats[i] = vec_to_quat(
            REFERENCE_AXIS,
            direction,
        )

    return ChainSample(
        positions=positions,
        directions=directions,
        quats=quats,
    )


def generate_dataset(
    num_samples=NUM_SAMPLES,
    num_sensors=NUM_SENSORS,
    link_length_cm=LINK_LENGTH_CM,
    bend_std_deg=BEND_STD_DEG,
    randomize_start_orientation=True,
    fixed_axes=(),
    seed=None,
):

    master_rng = np.random.default_rng(seed)
    
    # unique seed for each sample to ensure reproducibility
    sample_seeds = master_rng.integers(
        0,
        2**32 - 1,
        size=num_samples,
    )

    return [
        generate_chain(
            num_sensors=num_sensors,
            link_length_cm=link_length_cm,
            bend_std_deg=bend_std_deg,
            randomize_start_orientation=randomize_start_orientation,
            fixed_axes=fixed_axes,
            seed=int(sample_seed),
        )
        for sample_seed in sample_seeds
    ]

#Below functions are for the Data conversions to the format we want 

def positions_to_wide_row(positions):
    # Convert sensor positions into one wide row.
    row = {}

    for i in range(positions.shape[0]):
        row[f"{i + 1}_x"] = positions[i, 0]
        row[f"{i + 1}_y"] = positions[i, 1]
        row[f"{i + 1}_z"] = positions[i, 2]

    return row


def quats_to_wide_row(quats):
    # Convert sensor quaternions into one wide row.
    row = {}

    for i in range(quats.shape[0]):
        row[f"{i + 1}_w"] = quats[i, 0]
        row[f"{i + 1}_x"] = quats[i, 1]
        row[f"{i + 1}_y"] = quats[i, 2]
        row[f"{i + 1}_z"] = quats[i, 3]

    return row


def positions_to_cell_row(
    positions,
    decimals=6,
):
    # Convert sensor positions to one cell-style row.
    row = {}

    for i in range(positions.shape[0]):
        x, y, z = positions[i]

        row[f"sensor_{i + 1}"] = (
            f"[{x:.{decimals}f}, "
            f"{y:.{decimals}f}, "
            f"{z:.{decimals}f}]"
        )

    return row


def quats_to_cell_row(
    quats,
    decimals=6,
):
    # Convert sensor quaternions to one cell-style row.
    row = {}

    for i in range(quats.shape[0]):
        w, x, y, z = quats[i]

        row[f"sensor_{i + 1}"] = (
            f"[{w:.{decimals}f}, "
            f"{x:.{decimals}f}, "
            f"{y:.{decimals}f}, "
            f"{z:.{decimals}f}]"
        )

    return row


if __name__ == "__main__":
    samples = generate_dataset()

    # print(
    #     f"Generated {len(samples)} samples x "
    #     f"{samples[0].num_sensors} sensors "
    #     f"(fixed: {LINK_LENGTH_CM:g} cm links, "
    #     f"{BEND_STD_DEG:g} deg bend std-dev, "
    #     "random starting orientation per sample)\n"
    # )

    # Save generated positions to CSV
    # with open("positions.csv", "w", newline="") as file:
    #     writer = csv.DictWriter(
    #         file,
    #         fieldnames=positions_to_wide_row(samples[0].positions).keys(),
    #     )
    #     writer.writeheader()

    #     for sample in samples:
    #         writer.writerow(
    #             positions_to_wide_row(sample.positions)
    #         )

    # Save generated quaternions to CSV
    # with open("quaternions.csv", "w", newline="") as file:
    #     writer = csv.DictWriter(
    #         file,
    #         fieldnames=quats_to_wide_row(samples[0].quats).keys(),
    #     )
    #     writer.writeheader()

    #     for sample in samples:
    #         writer.writerow(
    #             quats_to_wide_row(sample.quats)
    #         )

    # print("Saved positions.csv")
    # print("Saved quaternions.csv")

    for sample_idx, sample in enumerate(samples):
        print(
            f"--- sample {sample_idx + 1} "
            f"of {len(samples)} ---"
        )
        print(
            f"{'sensor':>8} "
            f"{'x (cm)':>10} "
            f"{'y (cm)':>10} "
            f"{'z (cm)':>10}"
        )

        for i, position in enumerate(sample.positions):
            print(
                f"{'S' + str(i + 1):>8} "
                f"{position[0]:10.2f} "
                f"{position[1]:10.2f} "
                f"{position[2]:10.2f}"
            )

        print()