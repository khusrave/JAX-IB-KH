import dataclasses
from typing import Callable
import tree_math
from jax_ib.base import particle_class

# Note: The original file had many unused imports and classes.
# I have removed them for clarity. The core function is `forward_euler_updated`.

PyTreeState = any
TimeStepFn = Callable[[PyTreeState], PyTreeState]


@dataclasses.dataclass
class ExplicitNavierStokesODE_BCtime:
    """Describes the components of the Navier-Stokes time step."""
    explicit_terms: Callable
    pressure_projection: Callable
    update_BC: Callable
    Reserve_BC: Callable
    IBM_force: Callable
    Update_Position: Callable
    Pressure_Grad: Callable
    Calculate_Drag: Callable


@dataclasses.dataclass
class ButcherTableau_updated:
    a: list
    b: list
    c: list


def navier_stokes_rk_updated(
        tableau: ButcherTableau_updated,
        equation: ExplicitNavierStokesODE_BCtime,
        time_step: float,
) -> TimeStepFn:
    """Creates a forward Runge-Kutta time-stepper."""
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    P = tree_math.unwrap(equation.pressure_projection)
    M = tree_math.unwrap(equation.update_BC)
    IBM = tree_math.unwrap(equation.IBM_force)
    Update_Pos = tree_math.unwrap(equation.Update_Position)
    Drag_Calculation = tree_math.unwrap(equation.Calculate_Drag)

    b = tableau.b
    num_steps = len(b)

    @tree_math.wrap
    def step_fn(u0):
        # This is the main time-stepping logic. It takes the entire
        # `All_Variables` Pytree (renamed to u0 here) and updates it.

        # --- Helper functions to extract parts of the state ---
        the_velocity = lambda x: tree_math.Vector(x.tree.velocity)
        the_pressure = lambda x: x.tree.pressure

        # --- CORRECTED ATTRIBUTE NAMES ---
        the_Drag = lambda x: x.tree.intermediate_calcs  # Renamed from Drag
        the_step_counter = lambda x: x.tree.step_counter  # Renamed from Step_count
        # --- END CORRECTION ---

        the_MD_var = lambda x: x.tree.MD_var
        the_particles = lambda x: x.tree.particles

        # --- Deconstruct the state Pytree ---
        particles = the_particles(u0)
        ubc = tuple(v.bc for v in the_velocity(u0).tree)
        pressure = the_pressure(u0)
        Drag = the_Drag(u0)
        Step_count = the_step_counter(u0)
        MD_var = the_MD_var(u0)

        velocity_vec = the_velocity(u0)

        # --- Explicit forward Euler step ---
        k = F(velocity_vec)
        u_star = velocity_vec + dt * k

        # Create a temporary state object to pass to the IBM forcing function
        temp_all_variables = particle_class.All_Variables(
            particles, u_star.tree, pressure, Drag, Step_count, MD_var
        )

        # Calculate IBM force. This function returns an updated state AND the force.
        all_variables_after_force, force_density = IBM(temp_all_variables)

        # Run any intermediate calculations (like drag)
        all_variables_after_drag = Drag_Calculation(all_variables_after_force, dt)

        # Add the force to the velocity field
        u_star_star = u_star + dt * tree_math.Vector(force_density)

        # Project to enforce incompressibility
        # The pressure solve returns the pressure field, but P returns the updated velocity
        final_velocity_state = P(
            all_variables_after_drag._replace(velocity=u_star_star.tree)
        )

        # Update boundary conditions
        final_velocity_state = M(final_velocity_state)

        # Update particle positions based on the new fluid velocity
        final_state = Update_Pos(final_velocity_state, final_velocity_state.velocity, dt)

        # Increment the step counter
        final_state = final_state._replace(step_counter=final_state.step_counter + 1)

        return final_state

    return step_fn


def forward_euler_updated(
        equation: ExplicitNavierStokesODE_BCtime, time_step: float,
) -> TimeStepFn:
    """A wrapper for a simple forward Euler step."""
    return jax.named_call(
        navier_stokes_rk_updated(
            ButcherTableau_updated(a=[], b=[1], c=[0]),
            equation,
            time_step),
        name="forward_euler",
    )