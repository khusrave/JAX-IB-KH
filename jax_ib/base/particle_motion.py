import jax.numpy as jnp
import jax
from jax_ib.base import particle_class as pc
from jax_ib.base import grids, convolution_functions


# --- HELPER FUNCTION FOR DYNAMIC UPDATE ---

def update_mass_marker_positions(particles, dt):
    """
    Updates mass marker positions based on the pre-calculated point_force.
    Uses a simplified Euler integration step.
    """
    # This function is only called if the particle is dynamic.

    # The force on the mass marker is -F_total (Newton's 3rd Law).
    # This includes both penalty and tension forces.
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

    # Simplified position update: Y_new = Y_old + a * dt^2
    new_positions = particles.mass_marker_positions + acceleration * dt ** 2

    return new_positions


# --- MAIN UPDATE FUNCTION (MODIFIED) ---
# This is the primary function called by the solver.

def Update_particle_position_Multiple(all_variables, v, dt, force_density):
    """
    Top-level function to update particle positions.
    It checks if the particle is dynamic (has Kp) or kinematic.
    """
    particles = all_variables.particles

    # Check if the particle is dynamic (Kp > 0 is the key indicator).
    is_dynamic = hasattr(particles, 'Kp') and particles.Kp > 0

    if is_dynamic:
        # --- DYNAMIC UPDATE LOGIC ---
        # 1. Update fluid markers (X_m) by interpolating velocity
        xp, yp = particles.get_shape(all_variables.step_counter * dt)

        # We need a surface function to do the interpolation
        discrete_delta = lambda x, x0, w1: convolution_functions.delta_approx_logistjax(x, x0, w1)
        surface_fn = lambda field, xp, yp: convolution_functions.new_surf_fn(field, xp, yp, discrete_delta)

        velocity_at_surface = surface_fn(v, xp, yp)

        # Update the overall particle_center based on the average velocity of the markers
        new_particle_center = particles.particle_center + jnp.mean(velocity_at_surface, axis=0, keepdims=True) * dt

        # Store this new center in the particles object
        updated_particles_kinematic = particles._replace(particle_center=new_particle_center)

        # 2. Update mass markers (Y_m) using forces calculated in the previous step
        new_mass_positions = update_mass_marker_positions(updated_particles_kinematic, dt)

        final_updated_particles = updated_particles_kinematic._replace(mass_marker_positions=new_mass_positions)

        return all_variables._replace(particles=final_updated_particles)

    else:
        # --- KINEMATIC UPDATE LOGIC (Original Logic) ---
        current_t = v[0].bc.time_stamp
        particle_centers = particles.particle_center
        Displacement_EQ = particles.Displacement_EQ
        displacement_param = particles.displacement_param

        if Displacement_EQ is None:
            return all_variables  # If no kinematics, do nothing.

        New_eq = lambda t: Displacement_EQ(displacement_param, t)
        dx_dt = jax.jacrev(New_eq)

        U0 = dx_dt(current_t)
        new_particle_center = jnp.array([particle_centers[:, 0] + dt * U0[0],
                                         particle_centers[:, 1] + dt * U0[1]]).T

        updated_particles = particles._replace(particle_center=new_particle_center)

        return all_variables._replace(particles=updated_particles)


# --- ORIGINAL KINEMATIC HELPER (Kept for backward compatibility if needed elsewhere) ---
def Update_particle_position_Multiple_and_MD_Step(step_fn, all_variables, dt):
    # This is a specialized function from your original file.
    # We will keep it but it won't be used in our new dynamic simulation.
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

    # This creates a new particle object; we'll have to adapt it if we want to use it
    # with our new particle class definition.
    New_particles = pc.particle(
        particle_center=Newparticle_center,
        geometry_param=particles.geometry_param,
        displacement_param=displacement_param,
        rotation_param=particles.rotation_param,
        Grid=particles.Grid,
        shape=particles.shape,
        Displacement_EQ=Displacement_EQ,
        Rotation_EQ=particles.Rotation_EQ,
        # Defaulting new fields to None/0
        mass_marker_positions=None,
        point_force=None, sigma=0.0, Kp=0.0, particle_mass=0.0, g_vec=None
    )

    return pc.All_Variables(
        particles=New_particles,
        velocity=velocity,
        pressure=all_variables.pressure,
        intermediate_calcs=Drag,
        step_counter=all_variables.step_counter + 1,
        MD_var=MD_var
    )