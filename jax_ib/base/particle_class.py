import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
from jax_ib.base import grids

Array = Union[np.ndarray, jax.Array]
PyTree = Any

@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
  """Describes a 1D grid for particle parameterization."""
  shape: Tuple[int, ...]
  step: Tuple[float, ...]
  domain: Tuple[Tuple[float, float], ...]

  def __init__(
      self,
      shape: int, # Shape is an int for 1D
      step: Optional[Union[float, Sequence[float]]] = None,
      domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
  ):
    shape = (shape,) # Internally, shape is a tuple
    object.__setattr__(self, 'shape', shape)
    object.__setattr__(self, 'domain', domain)
    step = (domain[1] - domain[0]) / (shape[0]-1)
    object.__setattr__(self, 'step', (step,)) # Step is a tuple

  @property
  def ndim(self) -> int:
    return 1

  def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    if offset is None:
      offset = (0.5,) # Default offset for 1D
    return (self.domain[0] + (jnp.arange(self.shape[0]) + offset[0]) * self.step[0],)

  def mesh(self, offset: Optional[Sequence[float]] = None) -> Array:
      # For Grid1d, mesh is just the single axis array
      return self.axes(offset)[0]

@register_pytree_node_class
@dataclasses.dataclass
class particle:
    """A Pytree that contains all information about an immersed boundary."""
    # JIT-compatible fields (Arrays, numbers, etc.)
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    displacement_param: Optional[Sequence[Any]]
    rotation_param: Optional[Sequence[Any]]
    mass_marker_positions: Optional[jnp.ndarray]
    point_force: Optional[jnp.ndarray]
    sigma: float
    Kp: float
    particle_mass: float
    g_vec: Optional[jnp.ndarray]

    # Auxiliary (non-JIT) data
    Grid: Optional[Grid1d]
    shape: Callable
    Displacement_EQ: Optional[Callable]
    Rotation_EQ: Optional[Callable]

    def tree_flatten(self):
      """Specifies which attributes are JIT-compiled and which are static."""
      children = (self.particle_center, self.geometry_param, self.displacement_param, self.rotation_param,
                  self.mass_marker_positions, self.point_force,
                  self.sigma, self.Kp, self.particle_mass, self.g_vec)
      aux_data = (self.Grid, self.shape, self.Displacement_EQ, self.Rotation_EQ)
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Reconstructs the class from flattened data."""
       Grid, shape_fn, disp_eq, rot_eq = aux_data
       particle_center, geom_param, disp_param, rot_param, mass_pos, p_force, sigma, Kp, mass, g_vec = children
       return cls(particle_center, geom_param, disp_param, rot_param, mass_pos, p_force,
                  sigma, Kp, mass, g_vec, Grid, shape_fn, disp_eq, rot_eq)

    def get_shape(self, current_t):
        """Calculates the current positions of the boundary markers."""
        # For a dynamic body, the shape is defined by the particle_center only.
        # For a kinematic body, it's defined by the EQs.
        grid_p_mesh = self.Grid.mesh() if self.Grid is not None else None
        xp0, yp0 = self.shape(self.geometry_param[0], grid_p_mesh)

        if self.Rotation_EQ is not None and self.Displacement_EQ is not None:
            # Kinematic motion
            rotation_angle = self.Rotation_EQ(self.rotation_param, current_t)
            center_pos = self.Displacement_EQ(self.displacement_param, current_t)[0]
        else:
            # Dynamic motion (driven by forces, center updated by fluid)
            rotation_angle = 0.0
            center_pos = self.particle_center[0]

        xp = xp0 * jnp.cos(rotation_angle) - yp0 * jnp.sin(rotation_angle) + center_pos[0]
        yp = xp0 * jnp.sin(rotation_angle) + yp0 * jnp.cos(rotation_angle) + center_pos[1]
        return xp, yp

@register_pytree_node_class
@dataclasses.dataclass
class All_Variables:
    """A Pytree that contains all the simulation state variables."""
    particles: particle
    velocity: grids.GridVariableVector
    pressure: grids.GridVariable
    intermediate_calcs: Sequence[Any]
    step_counter: int
    MD_var: Any

    def tree_flatten(self):
      children = (self.particles, self.velocity, self.pressure,
                  self.intermediate_calcs, self.step_counter, self.MD_var)
      aux_data = None
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       return cls(*children)