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
PyTree = Any

@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
  shape: Tuple[int, ...]
  step: Tuple[float, ...]
  domain: Tuple[Tuple[float, float], ...]

  def __init__(
      self,
      shape: Sequence[int],
      step: Optional[Union[float, Sequence[float]]] = None,
      domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
  ):
    shape = shape
    object.__setattr__(self, 'shape', shape)
    object.__setattr__(self, 'domain', domain)
    step = (domain[1] - domain[0]) / (shape-1)
    object.__setattr__(self, 'step', step)

  @property
  def ndim(self) -> int:
    return 1

  @property
  def cell_center(self) -> Tuple[float, ...]:
    return self.ndim * (0.5,)

  def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    if offset is None:
      offset = self.cell_center
    if len(offset) != self.ndim:
      raise ValueError(f'unexpected offset length: {len(offset)} vs {self.ndim}')
    return self.domain[0] + jnp.arange(self.shape)*self.step

  def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    return self.axes(offset)

@register_pytree_node_class
@dataclasses.dataclass
class particle:
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
    Grid: Optional[Grid1d]
    shape: Callable
    Displacement_EQ: Optional[Callable]
    Rotation_EQ: Optional[Callable]

    def tree_flatten(self):
      children = (self.particle_center, self.geometry_param,
                  self.displacement_param, self.rotation_param,
                  self.mass_marker_positions, self.point_force,
                  self.sigma, self.Kp, self.particle_mass, self.g_vec)
      aux_data = (self.Grid, self.shape, self.Displacement_EQ, self.Rotation_EQ)
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       return cls(*children, *aux_data)

    @classmethod
    def create(cls, *, particle_center, geometry_param, shape_fn,
               displacement_param=None, rotation_param=None,
               displacement_eq=None, rotation_eq=None,
               Grid=None, mass_marker_positions=None,
               sigma=0.0, Kp=0.0, particle_mass=1.0, g_vec=None):
        return cls(particle_center=particle_center, geometry_param=geometry_param,
                   displacement_param=displacement_param, rotation_param=rotation_param,
                   mass_marker_positions=mass_marker_positions, point_force=None,
                   sigma=sigma, Kp=Kp, particle_mass=particle_mass, g_vec=g_vec,
                   Grid=Grid, shape=shape_fn,
                   Displacement_EQ=displacement_eq, Rotation_EQ=rotation_eq)

    def generate_grid(self):
        return self.Grid.mesh()

    def get_shape(self, current_t):
        xp0, yp0 = self.shape(self.geometry_param[0], self.Grid.mesh())

        if self.Rotation_EQ is not None and self.Displacement_EQ is not None:
            rotation_angle = self.Rotation_EQ(self.rotation_param, current_t)
            center_pos = self.Displacement_EQ(self.displacement_param, current_t)
        else: # Fallback for dynamic bodies where these are not defined
            rotation_angle = 0.0
            center_pos = self.particle_center[0]

        xp = (xp0) * jnp.cos(rotation_angle) - (yp0) * jnp.sin(rotation_angle) + center_pos[0]
        yp = (xp0) * jnp.sin(rotation_angle) + (yp0) * jnp.cos(rotation_angle) + center_pos[1]
        return xp, yp

@register_pytree_node_class
@dataclasses.dataclass
class All_Variables:
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

    @classmethod
    def create(cls, *, particles, velocity, pressure):
        return cls(particles=particles, velocity=velocity, pressure=pressure,
                   intermediate_calcs=[0], step_counter=0, MD_var=[0])