import dataclasses
import numbers
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
from jax_ib.base import grids


Array = Union[np.ndarray, jax.Array]
IntOrSequence = Union[int, Sequence[int]]

# There is currently no good way to indicate a jax "pytree" with arrays at its
# leaves. See https://jax.readthedocs.io/en/latest/jax.tree_util.html for more
# information about PyTrees and https://github.com/google/jax/issues/3340 for
# discussion of this issue.
PyTree = Any
@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
  """Describes the size and shape for an Arakawa C-Grid.

  See https://en.wikipedia.org/wiki/Arakawa_grids.

  This class describes domains that can be written as an outer-product of 1D
  grids. Along each dimension `i`:
  - `shape[i]` gives the whole number of grid cells on a single device.
  - `step[i]` is the width of each grid cell.
  - `(lower, upper) = domain[i]` gives the locations of lower and upper
    boundaries. The identity `upper - lower = step[i] * shape[i]` is enforced.
  """
  shape: Tuple[int, ...]
  step: Tuple[float, ...]
  domain: Tuple[Tuple[float, float], ...]

  def __init__(
      self,
      shape: Sequence[int],
      step: Optional[Union[float, Sequence[float]]] = None,
      domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
  ):
    """Construct a grid object."""
    shape = shape
    object.__setattr__(self, 'shape', shape)

 

    object.__setattr__(self, 'domain', domain)

    step = (domain[1] - domain[0]) / (shape-1) 
    object.__setattr__(self, 'step', step)

  @property
  def ndim(self) -> int:
    """Returns the number of dimensions of this grid."""
    return 1

  @property
  def cell_center(self) -> Tuple[float, ...]:
    """Offset at the center of each grid cell."""
    return self.ndim * (0.5,)



  def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """Returns a tuple of arrays containing the grid points along each axis.

    Args:
      offset: an optional sequence of length `ndim`. The grid will be shifted by
        `offset * self.step`.

    Returns:
      An tuple of `self.ndim` arrays. The jth return value has shape
      `[self.shape[j]]`.
    """
    if offset is None:
      offset = self.cell_center
    if len(offset) != self.ndim:
      raise ValueError(f'unexpected offset length: {len(offset)} vs '
                       f'{self.ndim}')

    return self.domain[0] + jnp.arange(self.shape)*self.step



  def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """Returns an tuple of arrays containing positions in each grid cell.

    Args:
      offset: an optional sequence of length `ndim`. The grid will be shifted by
        `offset * self.step`.

    Returns:
      An tuple of `self.ndim` arrays, each of shape `self.shape`. In 3
      dimensions, entry `self.mesh[n][i, j, k]` is the location of point
      `i, j, k` in dimension `n`.
    """
    
    return self.axes(offset)


# In jax_ib/base/particle_class.py

# ... (keep existing imports like dataclasses, jnp, etc.)
# ... (keep the Grid1d class as is)

@register_pytree_node_class
@dataclasses.dataclass
class particle:
    """A Pytree that contains all the information about the immersed boundary"""
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    displacement_param: Sequence[Any]
    rotation_param: Sequence[Any]

    # --- ADD NEW DYNAMIC FIELDS HERE ---
    mass_marker_positions: Optional[jnp.ndarray]
    point_force: Optional[jnp.ndarray]
    sigma: float
    Kp: float
    particle_mass: float
    g_vec: Optional[jnp.ndarray]

    # --- AUXILIARY (NON-JIT) DATA ---
    Grid: Grid1d
    shape: Callable
    Displacement_EQ: Callable
    Rotation_EQ: Callable

    def tree_flatten(self):
        """Returns flattening recipe for GridVariable JAX pytree."""
        # --- UPDATE THE CHILDREN TUPLE WITH ALL JIT-COMPATIBLE FIELDS ---
        children = (self.particle_center, self.geometry_param,
                    self.displacement_param, self.rotation_param,
                    self.mass_marker_positions, self.point_force,
                    self.sigma, self.Kp, self.particle_mass, self.g_vec)

        aux_data = (self.Grid, self.shape, self.Displacement_EQ, self.Rotation_EQ)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Returns unflattening recipe for GridVariable JAX pytree."""
        # This now correctly unpacks all children and aux_data
        return cls(*children, *aux_data)

    # --- ADD A CUSTOM CONSTRUCTOR FOR EASIER USE ---
    # This allows creating a particle without providing all arguments every time.
    @classmethod
    def create(cls,
               particle_center,
               geometry_param,
               shape_fn,
               displacement_param=None,
               rotation_param=None,
               displacement_eq=None,
               rotation_eq=None,
               Grid=None,  # <-- CORRECTED: Argument is now 'Grid' with a capital G
               mass_marker_positions=None,
               sigma=0.0,
               Kp=0.0,
               particle_mass=1.0,
               g_vec=None):
        """A user-friendly constructor for the particle class."""
        return cls(
            particle_center=particle_center,
            geometry_param=geometry_param,
            displacement_param=displacement_param,
            rotation_param=rotation_param,
            mass_marker_positions=mass_marker_positions,
            point_force=None,
            sigma=sigma,
            Kp=Kp,
            particle_mass=particle_mass,
            g_vec=g_vec,
            Grid=Grid,  # <-- CORRECTED: Pass the 'Grid' argument here
            shape=shape_fn,
            Displacement_EQ=displacement_eq,
            Rotation_EQ=rotation_eq
        )

    def generate_grid(self):
        return self.Grid.mesh()

    def get_shape(self, current_t):
        """Calculates the current positions of the boundary markers."""
        # This is a key function. For dynamic bodies, it should return the
        # current positions stored in the state. For kinematic bodies, it
        # calculates them based on the prescribed motion.
        # The library's `Update_particle_position_Multiple` function modifies
        # the particle_center and rotation to achieve this.

        xp0, yp0 = self.shape(self.geometry_param, self.Grid.mesh())  # Using self.shape and self.Grid

        # NOTE: For a fully dynamic body, we would eventually remove this kinematic part.
        rotation_angle = self.Rotation_EQ([self.rotation_param], current_t) if self.Rotation_EQ else 0.0
        center_pos = self.Displacement_EQ([self.displacement_param],
                                          current_t) if self.Displacement_EQ else self.particle_center

        xp = (xp0) * jnp.cos(rotation_angle) - (yp0) * jnp.sin(rotation_angle) + center_pos[0]
        yp = (xp0) * jnp.sin(rotation_angle) + (yp0) * jnp.cos(rotation_angle) + center_pos[1]

        return xp, yp


@register_pytree_node_class
@dataclasses.dataclass
class All_Variables:
    """A Pytree that contains all the information for an immersed boundary simulation."""
    particles: particle
    velocity: grids.GridVariableVector
    pressure: grids.GridVariable
    intermediate_calcs: Sequence[Any]
    step_counter: int
    MD_var: Any

    def tree_flatten(self):
        """Returns flattening recipe for GridVariable JAX pytree."""
        children = (self.particles, self.velocity, self.pressure,
                    self.intermediate_calcs, self.step_counter, self.MD_var)

        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Returns unflattening recipe for GridVariable JAX pytree."""
        return cls(*children)

    @classmethod
    def create(cls, *, particles, velocity, pressure):
        """Creates a new All_Variables state."""
        return cls(
            particles=particles,
            velocity=velocity,
            pressure=pressure,
            intermediate_calcs=[0],
            step_counter=0,
            MD_var=[0],
        )
    
@register_pytree_node_class
@dataclasses.dataclass
class particle_lista: # SEQUENCE OF VARIABLES MATTER !
    particles: Sequence[particle,]

    
    def generate_grid(self):
        
        return np.stack([grid.mesh() for grid in self.Grid])
       
    def calc_Rtheta(self):
      return self.shape(self.geometry_param,self.Grid) 
    
    def tree_flatten(self):
      """Returns flattening recipe for GridVariable JAX pytree."""
      children = (*self.particles,)
      aux_data = None
      return children,aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Returns unflattening recipe for GridVariable JAX pytree."""
       return cls(*children)
    
    

    
    
