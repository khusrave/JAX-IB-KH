import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax import debug as jax_debug


# --- HELPER FUNCTIONS ---

def calculate_tension_force(xp, yp, sigma):
    """Calculates the surface tension force (component-wise)."""
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL ** 2 + dyL ** 2) + 1e-9
    l_hat_x = dxL / dS
    l_hat_y = dyL / dS
    l_hat_x_prev = jnp.roll(l_hat_x, 1)
    l_hat_y_prev = jnp.roll(l_hat_y, 1)
    force_x = sigma * (l_hat_x - l_hat_x_prev)  # NOTE: This is the expanding version from your notebook
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return force_x, force_y


def calculate_penalty_force(mass_marker_positions, fluid_marker_positions, Kp):
    """Calculates the penalty (spring) force between mass and fluid markers."""
    force = Kp * (mass_marker_positions - fluid_marker_positions)
    return force


# --- MODIFIED CORE FORCING FUNCTION ---
# This function is the only one we need to modify to add the new force.
def IBM_force_GENERAL(
        field, Xi, particle_center, geom_param, Grid_p, shape_fn,
        discrete_fn, surface_fn, dx_dt, domega_dt, rotation, dt,
        sigma=0.0,  # Keep sigma from the old signature for now
        # ADD new arguments that we will get from the `particles` object
        mass_marker_positions=None,
        Kp=0.0,
        particle_mass=1.0,
        g_vec=None
):
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    current_t = field.bc.time_stamp

    # This part calculates the current fluid marker positions (X_m) from kinematics
    xp0, yp0 = shape_fn(geom_param, Grid_p)
    xp = (xp0) * jnp.cos(rotation(current_t)) - (yp0) * jnp.sin(rotation(current_t)) + particle_center[0]
    yp = (xp0) * jnp.sin(rotation(current_t)) + (yp0) * jnp.cos(rotation(current_t)) + particle_center[1]
    fluid_positions = jnp.stack([xp, yp], axis=-1)

    # Calculate segment lengths (dS) for density conversion (unchanged)
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL ** 2 + dyL ** 2) + 1e-9

    # --- START OF MODIFICATIONS ---

    # Initialize point forces as zero vectors
    total_point_force = jnp.zeros_like(fluid_positions)

    # 1. Calculate and add tension force (if sigma > 0)
    if sigma > 0.0:
        tension_force_x, tension_force_y = calculate_tension_force(xp, yp, sigma)
        total_point_force += jnp.stack([tension_force_x, tension_force_y], axis=-1)

    # 2. Calculate and add penalty force (if Kp > 0)
    if Kp > 0.0 and mass_marker_positions is not None:
        penalty_force = calculate_penalty_force(mass_marker_positions, fluid_positions, Kp)
        total_point_force += penalty_force

    # For a purely dynamic body, the "direct forcing" (from prescribed motion) is zero.
    # The total force density on the fluid is the reaction to our calculated dynamic forces.
    # The reaction force is -total_point_force.

    if Xi == 0:
        point_force_component = -total_point_force[:, 0]
    else:  # Xi == 1
        point_force_component = -total_point_force[:, 1]

    # Convert point force to force density by dividing by segment length
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


# --- WRAPPER FUNCTIONS (Modified to pass new state and parameters) ---

def IBM_Multiple_NEW(field, Xi, particles, discrete_fn, surface_fn, dt, sigma=0.0):
    Grid_p = particles.generate_grid()
    shape_fn = particles.shape
    Displacement_EQ = particles.Displacement_EQ
    Rotation_EQ = particles.Rotation_EQ
    Nparticles = len(particles.particle_center)
    particle_center = particles.particle_center
    geom_param = particles.geometry_param
    displacement_param = particles.displacement_param
    rotation_param = particles.rotation_param
    force = jnp.zeros_like(field.data)

    # --- Extract dynamic properties from the particle object ---
    # These will be passed down to IBM_force_GENERAL
    mass_marker_positions = particles.mass_marker_positions
    Kp = particles.Kp
    particle_mass = particles.particle_mass
    g_vec = particles.g_vec
    # The 'sigma' from the function signature overrides the particle's sigma for now.

    for i in range(Nparticles):
        Xc = lambda t: Displacement_EQ([displacement_param[i]], t) if Displacement_EQ else particle_center[i]
        rotation = lambda t: Rotation_EQ([rotation_param[i]], t) if Rotation_EQ else 0.0
        dx_dt = jax.jacrev(Xc)
        domega_dt = jax.jacrev(rotation)

        force += IBM_force_GENERAL(
            field, Xi, particle_center[i], geom_param[i], Grid_p, shape_fn,
            discrete_fn, surface_fn, dx_dt, domega_dt, rotation, dt,
            sigma=sigma,  # <-- using sigma from the argument list
            mass_marker_positions=mass_marker_positions,
            Kp=Kp,
            particle_mass=particle_mass,
            g_vec=g_vec
        )
    return grids.GridArray(force, field.offset, field.grid)


def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, surface_fn, dt, sigma=0.0):
    velocity = all_variables.velocity
    particles = all_variables.particles

    # If the particle object has a sigma, use it. Otherwise use the one passed as an argument.
    # This maintains backward compatibility.
    sigma_to_use = particles.sigma if hasattr(particles, 'sigma') and particles.sigma > 0 else sigma

    axis = [0, 1]
    ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(
        field, Xi, particles, discrete_fn, surface_fn, dt, sigma=sigma_to_use
    )

    return tuple(grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))