"""9/18/27/36D observations in the author-provided 1015appendix.pdf order.

Feature identities and indices come from the appendix. Normalization, pending
cluster membership, guidance magnitude and confidence formulas are explicit
implementation choices; see docs/appendix_alignment.md.
"""
import numpy as np


FEATURES = {
    9: ["relative_x", "relative_y", "relative_z", "distance", "capture", "vx", "vy", "vz", "time"],
    18: ["auv_x", "auv_y", "auv_z", "target_x", "target_y", "target_z", "distance", "capture",
         "vx", "vy", "vz", "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch", "time", "surface_distance", "seabed_distance"],
    27: ["auv_x", "auv_y", "auv_z", "target_x", "target_y", "target_z", "distance", "capture",
         "vx", "vy", "vz", "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch", "time", "current_x", "current_y", "current_z",
         "obstacle1_dx", "obstacle1_dy", "obstacle1_dz", "obstacle1_radius", "obstacle2_dx", "obstacle2_dy", "obstacle2_dz", "obstacle2_radius"],
    36: ["auv_x", "auv_y", "auv_z", "target_x", "target_y", "target_z", "relative_x", "relative_y", "relative_z", "distance", "capture",
         "vx", "vy", "vz", "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch", "yaw_rate", "time", "current_x", "current_y", "current_z",
         "obstacle1_dx", "obstacle1_dy", "obstacle1_dz", "obstacle1_radius", "obstacle2_dx", "obstacle2_dy", "obstacle2_dz", "obstacle2_radius",
         "vehicle_noise", "biological_noise", "geological_noise", "turbulence_noise", "perception_confidence"],
}


def appendix_observation(env):
    position = env.agent.position
    # Appendix describes the target cluster centroid. Acquired targets leave the pending cluster.
    pending = env.targets[~env.captured]
    target = pending.mean(axis=0) if len(pending) else env.targets.mean(axis=0)
    relative = target - position
    distance = float(np.linalg.norm(relative))
    f = env.field_size
    signed = lambda x, bound: np.clip((np.asarray(x) / max(bound, 1e-10) + 1) / 2, 0, 1)
    flags = [float(distance < env.target_radius)]
    time = [min(env.current_step / env.max_step, 1)]
    velocity = signed(env.agent.velocity, env.agent.max_speed)
    relative_scaled = signed(relative, 2 * f)
    distance_scaled = [np.clip(distance / (2 * np.sqrt(3) * f), 0, 1)]
    attitude = signed([np.sin(env.agent.theta), np.cos(env.agent.theta), np.sin(env.agent.phi), np.cos(env.agent.phi)], 1)
    absolute = np.concatenate([signed(position, f), signed(target, f), distance_scaled, flags])
    dim = env.layout.observation_dim
    if dim == 9:
        # Body-local displacement: inverse yaw/pitch rotation of inertial delta.
        yaw, pitch = env.agent.theta, env.agent.phi
        rotation = np.array([[np.cos(yaw)*np.cos(pitch), -np.sin(yaw), -np.cos(yaw)*np.sin(pitch)],
                             [np.sin(yaw)*np.cos(pitch), np.cos(yaw), -np.sin(yaw)*np.sin(pitch)],
                             [np.sin(pitch), 0, np.cos(pitch)]])
        # sqrt(3) bound accommodates rotation of a cube's displacement vector.
        local = signed(rotation.T @ relative, 2 * np.sqrt(3) * f)
        return np.concatenate([local, distance_scaled, flags, velocity, time]).astype(np.float32)
    if dim == 18:
        depths = np.clip([(f - position[2]) / (2*f), (f + position[2]) / (2*f)], 0, 1)
        return np.concatenate([absolute, velocity, attitude, time, depths]).astype(np.float32)
    nearest = np.argsort(np.linalg.norm(env.obstacles - position, axis=1))[:2]
    obstacles = []
    for slot in range(2):
        if slot < len(nearest):
            offset = env.obstacles[nearest[slot]] - position
            obstacles.extend([*signed(offset, 2*f), np.clip(env.obstacle_radius / f, 0, 1)])
        else:
            # Neutral offset and zero radius represent a missing obstacle slot.
            obstacles.extend([0.5, 0.5, 0.5, 0.0])
    barrier = np.concatenate([signed(env.current, env.current_strength), obstacles])
    if dim == 27:
        return np.concatenate([absolute, velocity, attitude, time, barrier]).astype(np.float32)
    spatial = np.concatenate([signed(position, f), signed(target, f), relative_scaled, distance_scaled, flags])
    noise = 0.5 + np.arctan(env.noise / max(env.noise_std, 1e-8)) / np.pi
    # Appendix names the confidence feature but supplies no formula; explicit bounded choice.
    confidence = [1 / (1 + np.mean(np.abs(env.noise)) / max(env.noise_std, 1e-8))]
    return np.concatenate([spatial, velocity, attitude, signed([env.yaw_rate], env.max_turn), time,
                           barrier, noise, confidence]).astype(np.float32)
