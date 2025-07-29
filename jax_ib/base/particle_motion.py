import jax.numpy as jnp
import jax
from jax_ib.base import particle_class as pc
from jax_ib.base import grids  # Make sure this is imported


# --- ADD THIS NEW HELPER FUNCTION ---
def update_mass_marker_positions(particles, dt):
    """
    Updates the position of mass markers based on Sustiel & Grier, Eq. (5).
    Uses a simple Symplectic Euler update. This version works on a single particle object.
    """
    # The force on the mass marker is -F_penalty. We assume the penalty force
    # is the dominant part of `point_force` for this update.
    # A more complex model could split the forces.
    force_on_mass_marker = -particles.point_force

    mass_per_marker = particles.particle_mass / particles.mass_marker_positions.shape[0]

    # Add gravity if it exists
    if particles.g_vec is not None:
        gravity_force = mass_per_marker * particles.g_vec
        total_force = force_on_mass_marker + gravity_force
    else:
        total_force = force_on_mass_marker

    # Calculate acceleration: a = F / m
    acceleration = total_force / mass_per_marker

    # Simplified position update (approximates Y_new = Y_old + a * dt^2)
    # This is a basic Euler step. More advanced integrators could be used here.
    new_positions = particles.mass_marker_positions + acceleration * dt ** 2

    return new_positions


# --- ADD THIS NEW MAIN UPDATE FUNCTION FOR DYNAMIC BODIES ---
def Update_particle_state_dynamic(all_variables, v, dt, force_density, surface_fn):
    """
    Updates the full state of a dynamic particle for the next time step.
    1. Updates fluid marker positions (X_m) by interpolating fluid velocity.
    2. Updates mass marker positions (Y_m) by integrating the penalty force.
    """
    particles = all_variables.particles

    # --- 1. Update fluid markers (X_m) ---
    # This moves the massless boundary points with the local fluid velocity.
    xp, yp = particles.get_shape(all_variables.step_counter * dt)
    velocity_at_surface = surface_fn(v, xp, yp)  # Interpolate fluid velocity

    # Update the particle_center based on the average velocity of the markers
    # This makes the entire object translate with the fluid.
    new_particle_center = particles.particle_center + jnp.mean(velocity_at_surface, axis=0) * dt

    # Store this new center in the particles object
    updated_particles_kinematic = particles._replace(particle_center=new_particle_center)

    # --- 2. Update mass markers (Y_m) using our new function ---
    # This uses the `point_force` that was calculated and stored in the IBM_force step.
    new_mass_positions = update_mass_marker_positions(updated_particles_kinematic, dt)

    # Store the new mass marker positions back into the particle object.
    final_updated_particles = updated_particles_kinematic._replace(mass_marker_positions=new_mass_positions)

    # Return the fully updated particles object to be stored in the state Pytree.
    return final_updated_particles


# --- ORIGINAL KINEMATIC FUNCTIONS (Unchanged, for backward compatibility) ---

def Update_particle_position_Multiple_and_MD_Step(step_fn, all_variables, dt):
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t = velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param
    New_eq = lambda t: Displacement_EQ(displacement_param, t)
    dx_dt = jax.jacrev(New_eq)

    MD_var = step_fn(all_variables)

    U0 = dx_dt(current_t)
    Newparticle_center = jnp.array([particle_centers[:, 0] + dt * U0[0], particle_centers[:, 1] + dt * U0[1]]).T

    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    rotation_param = particles.rotation_param

    # Re-create the particle object with the updated kinematic state
    # NOTE: This does not include the new dynamic fields. This function is for kinematic-only particles.
    New_particles = pc.particle(Newparticle_center, param_geometry, displacement_param, rotation_param, mygrids,
                                shape_fn, Displacement_EQ, particles.Rotation_EQ)

    return pc.All_Variables(New_particles, velocity, pressure, Drag, Step_count, MD_var)


def Update_particle_position_Multiple(particles, v, dt, force_density):
    # This function from the original library seems to update particle_center
    # based on prescribed kinematics. We are creating a new dynamic version.
    # To avoid errors, let's keep the original function signature but modify it
    # slightly to pass the whole state, which is what your notebook seems to do.
    all_variables = particles  # Assuming the 'particles' argument is actually the whole state Pytree

    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t = velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param

    # Handle cases where kinematic equations are not defined
    if Displacement_EQ is None:
        return particles  # If no kinematics, do nothing.

    New_eq = lambda t: Displacement_EQ(displacement_param, t)
    dx_dt = jax.jacrev(New_eq)

    U0 = dx_dt(current_t)
    Newparticle_center = jnp.array([particle_centers[:, 0] + dt * U0[0], particle_centers[:, 1] + dt * U0[1]]).T

    # Re-create the particle object with the updated kinematic state
    updated_particles = particles._replace(particle_center=Newparticle_center)

    return updated_particles