import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax_ib.base import particle_class as pc
from functools import partial


# --- HELPER FUNCTIONS FOR DYNAMIC FORCES ---

def calculate_tension_force_vectorized(positions, sigma):
    """
    Calculates surface tension force (vectorized).
    NOTE: Using the expanding-force sign convention from your notebook.
    For a contracting force, use: sigma * (l_hat_rolled_backward - l_hat)
    """
    dxL = jnp.roll(positions[:, 0], -1) - positions[:, 0]
    dyL = jnp.roll(positions[:, 1], -1) - positions[:, 1]
    dS = jnp.sqrt(dxL ** 2 + dyL ** 2) + 1e-9
    l_hat_x = dxL / dS
    l_hat_y = dyL / dS
    l_hat_x_prev = jnp.roll(l_hat_x, 1)
    l_hat_y_prev = jnp.roll(l_hat_y, 1)
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return jnp.stack([force_x, force_y], axis=-1)


def calculate_penalty_force(mass_marker_positions, fluid_marker_positions, Kp):
    """Calculates the penalty (spring) force between mass and fluid markers."""
    return Kp * (mass_marker_positions - fluid_marker_positions)


# --- HELPER FUNCTION FOR SPREADING FORCE TO GRID ---

def _point_force_to_force_density(xp, yp, F, grid, offset, discrete_fn):
    """Spreads a component of point forces to the Eulerian grid."""
    X, Y = grid.mesh(offset)
    # The delta function is evaluated for all grid points vs. all particle points
    delta_values = discrete_fn(X[..., None], Y[..., None], xp[None, None, ...], yp[None, None, ...])
    # The force density at each grid point is the sum of contributions from all particles
    force_density = jnp.einsum('ijk,k->ij', delta_values, F)
    return force_density


def calc_force_density(particles, v, discrete_fn, surface_fn, dt):
    """Computes the force density tuple (fx, fy) for a single particle object."""
    xp, yp = particles.get_shape(v[0].bc.time_stamp)
    F = particles.point_force

    # The force on the fluid is the reaction force (-F)
    force_density_x = _point_force_to_force_density(xp, yp, -F[:, 0], v[0].grid, v[0].offset, discrete_fn)
    force_density_y = _point_force_to_force_density(xp, yp, -F[:, 1], v[1].grid, v[1].offset, discrete_fn)

    return (grids.GridArray(force_density_x, v[0].offset, v[0].grid),
            grids.GridArray(force_density_y, v[1].offset, v[1].grid))


# --- MAIN FORCING FUNCTION (MODIFIED) ---
# This is the primary function called by the solver. It now handles both kinematic and dynamic cases.

def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, surface_fn, dt, sigma=0.0):
    """
    Top-level function to compute IBM forces.
    It checks if the particle is dynamic (has sigma or Kp) or kinematic.
    """
    velocity = all_variables.velocity
    particles = all_variables.particles

    # --- DYNAMIC FORCE CALCULATION ---
    # Check if the particle has dynamic properties.
    is_dynamic = (hasattr(particles, 'sigma') and particles.sigma > 0) or \
                 (hasattr(particles, 'Kp') and particles.Kp > 0)

    if is_dynamic:
        xp, yp = particles.get_shape(all_variables.step_counter * dt)
        fluid_positions = jnp.stack([xp, yp], axis=-1)

        total_point_force = jnp.zeros_like(fluid_positions)

        if particles.sigma > 0:
            total_point_force += calculate_tension_force_vectorized(fluid_positions, particles.sigma)

        if particles.Kp > 0 and particles.mass_marker_positions is not None:
            total_point_force += calculate_penalty_force(particles.mass_marker_positions, fluid_positions, particles.Kp)

        updated_particles = particles._replace(point_force=total_point_force)
        all_variables = all_variables._replace(particles=updated_particles)

        force_density_tuple = calc_force_density(updated_particles, velocity, discrete_fn, surface_fn, dt)
        force_density_vars = tuple(grids.GridVariable(array, v.bc) for array, v in zip(force_density_tuple, velocity))

        return all_variables, force_density_vars

    # --- KINEMATIC FORCE CALCULATION (Original Logic) ---
    else:
        # This part handles the original kinematic behavior of the library.
        axis = [0, 1]
        ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(
            field, Xi, particles, discrete_fn, surface_fn, dt, sigma=sigma
        )
        force_density_vars = tuple(
            grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))
        return all_variables, force_density_vars


# --- ORIGINAL KINEMATIC HELPER FUNCTIONS (Kept for backward compatibility) ---

def IBM_Multiple_NEW(field, Xi, particles, discrete_fn, surface_fn, dt, sigma=0.0):
    Grid_p = particles.Grid.mesh()  # Simplified this call
    shape_fn = particles.shape
    Displacement_EQ = particles.Displacement_EQ
    Rotation_EQ = particles.Rotation_EQ
    Nparticles = len(particles.particle_center)
    particle_center = particles.particle_center
    geom_param = particles.geometry_param
    displacement_param = particles.displacement_param
    rotation_param = particles.rotation_param
    force = jnp.zeros_like(field.data)

    for i in range(Nparticles):
        Xc = lambda t: Displacement_EQ([displacement_param[i]], t)
        rotation = lambda t: Rotation_EQ([rotation_param[i]], t)
        dx_dt = jax.jacrev(Xc)
        domega_dt = jax.jacrev(rotation)

        # NOTE: This calls the original IBM_force_GENERAL logic
        force += IBM_force_GENERAL(
            field, Xi, particle_center[i], geom_param[i], Grid_p, shape_fn,
            discrete_fn, surface_fn, dx_dt, domega_dt, rotation, dt, sigma
        )
    return grids.GridArray(force, field.offset, field.grid)


def IBM_force_GENERAL(
        field, Xi, particle_center, geom_param, Grid_p, shape_fn, discrete_fn, surface_fn,
        dx_dt, domega_dt, rotation, dt, sigma=0.0
):
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    current_t = field.bc.time_stamp
    xp0, yp0 = shape_fn(geom_param, Grid_p)
    xp = (xp0) * jnp.cos(rotation(current_t)) - (yp0) * jnp.sin(rotation(current_t)) + particle_center[0]
    yp = (xp0) * jnp.sin(rotation(current_t)) + (yp0) * jnp.cos(rotation(current_t)) + particle_center[1]

    velocity_at_surface = surface_fn(field, xp, yp)

    if Xi == 0:
        position_r = -(yp - particle_center[1])
    else:  # Xi == 1
        position_r = (xp - particle_center[0])

    U0 = dx_dt(current_t)
    Omega = domega_dt(current_t)
    UP = U0[Xi] + Omega * position_r

    direct_force_density = (UP - velocity_at_surface) / dt

    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL ** 2 + dyL ** 2) + 1e-9

    total_force_density = direct_force_density
    if sigma > 0.0:
        tension_force_x, tension_force_y = calculate_tension_force(xp, yp, sigma)
        if Xi == 0:
            total_force_density -= (tension_force_x / dS)
        else:
            total_force_density -= (tension_force_y / dS)

    # Simplified spreading without pmap for clarity and compatibility
    delta_values = discrete_fn(X[..., None], Y[..., None], xp[None, None, ...], yp[None, None, ...])
    force = jnp.einsum('ijk,k->ij', delta_values, total_force_density * dS)

    return force