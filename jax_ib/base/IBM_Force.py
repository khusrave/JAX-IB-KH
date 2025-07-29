import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax import debug as jax_debug


# --- HELPER FUNCTIONS FOR FORCES ---

def calculate_tension_force(xp, yp, sigma):
    """
    Calculates the surface tension force on each Lagrangian marker.
    """
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL ** 2 + dyL ** 2) + 1e-9
    l_hat_x = dxL / dS
    l_hat_y = dyL / dS
    l_hat_x_prev = jnp.roll(l_hat_x, 1)
    l_hat_y_prev = jnp.roll(l_hat_y, 1)
    # Note: This is the expanding-force version.
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return force_x, force_y


def calculate_penalty_force(mass_marker_positions, fluid_marker_positions, Kp):
    """
    Calculates the penalty force (spring force) based on Sustiel & Grier, Eq. (4).
    F_m = Kp * (Y_m - X_m)
    Returns a (N, 2) array of force vectors.
    """
    force = Kp * (mass_marker_positions - fluid_marker_positions)
    return force


# --- INTEGRATION HELPER FUNCTIONS (Unchanged) ---

def integrate_trapz(integrand, dx, dy):
    return jnp.trapz(jnp.trapz(integrand, dx=dx), dx=dy)


def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    return integrate_trapz(field.data, dxEUL, dyEUL)


# --- CORE FORCING FUNCTION (Modified for Dynamic Forces) ---

def IBM_force_GENERAL(
        field, Xi, particle_center, geom_param, Grid_p, shape_fn,
        discrete_fn, surface_fn, dx_dt, domega_dt, rotation, dt,
        # Add new arguments to receive dynamic properties
        mass_marker_positions=None,
        sigma=0.0,
        Kp=0.0
):
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    current_t = field.bc.time_stamp

    # This part calculates the current fluid marker positions (X_m)
    xp0, yp0 = shape_fn(geom_param, Grid_p)
    xp = (xp0) * jnp.cos(rotation(current_t)) - (yp0) * jnp.sin(rotation(current_t)) + particle_center[0]
    yp = (xp0) * jnp.sin(rotation(current_t)) + (yp0) * jnp.cos(rotation(current_t)) + particle_center[1]
    fluid_positions = jnp.stack([xp, yp], axis=-1)

    # --- START OF MODIFICATIONS ---
    # We will now calculate a total point force from dynamic sources
    # and use it INSTEAD of the kinematic direct forcing.

    # Initialize total point force array (N, 2)
    total_point_force = jnp.zeros_like(fluid_positions)

    # 1. Calculate and add surface tension force
    if sigma is not None and sigma > 0.0:
        tension_force_x, tension_force_y = calculate_tension_force(xp, yp, sigma)
        total_point_force += jnp.stack([tension_force_x, tension_force_y], axis=-1)

    # 2. Calculate and add penalty force
    if Kp is not None and Kp > 0.0 and mass_marker_positions is not None:
        penalty_force = calculate_penalty_force(mass_marker_positions, fluid_positions, Kp)
        total_point_force += penalty_force

    # The force on the fluid is the reaction force (-F_total).
    # We select the component (x or y) for this specific call.
    if Xi == 0:
        point_force_component = -total_point_force[:, 0]
    else:  # Xi == 1
        point_force_component = -total_point_force[:, 1]

    # Calculate segment lengths dS for density conversion (unchanged)
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL ** 2 + dyL ** 2) + 1e-9

    # Convert the point force component into a force density
    total_force_density = point_force_component / dS

    # --- END OF MODIFICATIONS ---

    # Spreading logic remains the same
    def calc_force(F, xp, yp, dxi, dyi, dss):
        return F * discrete_fn(jnp.sqrt((xp - X) ** 2 + (yp - Y) ** 2), 0, grid.step[0]) * dss

    def foo(tree_arg):
        F, xp, yp, dxi, dyi, dss = tree_arg
        return calc_force(F, xp, yp, dxi, dyi, dss)

    def foo_pmap(tree_arg):
        return jnp.sum(jax.vmap(foo, in_axes=1)(tree_arg), axis=0)

    divider = jax.device_count()
    n = len(xp) // divider
    mapped = []
    for i in range(divider):
        mapped.append([total_force_density[i * n:(i + 1) * n], xp[i * n:(i + 1) * n], yp[i * n:(i + 1) * n],
                       dxL[i * n:(i + 1) * n], dyL[i * n:(i + 1) * n], dS[i * n:(i + 1) * n]])

    return jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)), axis=0)


# --- WRAPPER FUNCTIONS (Modified to handle new state and parameters) ---

def IBM_Multiple_NEW(
        field, Xi, particles, discrete_fn, surface_fn, dt
):
    Grid_p = particles.generate_grid()
    shape_fn = particles.shape
    Displacement_EQ = particles.Displacement_EQ
    Rotation_EQ = particles.Rotation_EQ
    Nparticles = len(particles.particle_center)
    particle_center = particles.particle_center
    geom_param = particles.geometry_param
    displacement_param = particles.displacement_param
    rotation_param = particles.rotation_param

    # Extract dynamic properties from the particle object
    mass_marker_positions = particles.mass_marker_positions
    sigma = particles.sigma
    Kp = particles.Kp

    force = jnp.zeros_like(field.data)
    for i in range(Nparticles):
        # NOTE: For a fully dynamic simulation, the kinematic part (Xc, rotation)
        # should be turned off (e.g., params set to zero). The positions xp, yp
        # would then come from interpolating the fluid velocity.
        # For now, we keep the structure but our new forces will dominate.
        Xc = lambda t: Displacement_EQ([displacement_param[i]], t)
        rotation = lambda t: Rotation_EQ([rotation_param[i]], t)
        dx_dt = jax.jacrev(Xc)
        domega_dt = jax.jacrev(rotation)

        force += IBM_force_GENERAL(
            field, Xi, particle_center[i], geom_param[i], Grid_p, shape_fn,
            discrete_fn, surface_fn, dx_dt, domega_dt, rotation, dt,
            mass_marker_positions=mass_marker_positions,  # Pass state
            sigma=sigma, Kp=Kp  # Pass parameters
        )
    return grids.GridArray(force, field.offset, field.grid)


def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, surface_fn, dt, sigma=0.0):
    # This function is now the kinematic entry point.
    # To use our dynamic forces, we will call a different function from the notebook.
    velocity = all_variables.velocity
    particles = all_variables.particles
    # To use the dynamic version, we need to pass sigma, Kp, etc., from the particle object.
    # We will modify this to read from the particle object if available.
    if hasattr(particles, 'sigma'):
        sigma = particles.sigma  # Override with value from particle object

    axis = [0, 1]
    ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(
        field, Xi, all_variables.particles, discrete_fn, surface_fn, dt
    )

    return tuple(grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))