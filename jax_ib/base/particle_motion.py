from jax_ib.base import particle_class as pc
import jax
import jax.numpy as jnp

from jax_ib.base.interpolation import point_interpolation


def Update_particle_position_Multiple_and_MD_Step(step_fn,all_variables,dt):
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t =velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param    
    New_eq = lambda t:Displacement_EQ(displacement_param,t)
    dx_dt = jax.jacrev(New_eq)

    
    #MD_var = step_fn(MD_var)
    
    U0 =dx_dt(current_t)
    #print(U0)
    Newparticle_center = jnp.array([particle_centers[:,0]+dt*U0[0],particle_centers[:,1]+dt*U0[1]]).T
    #print(Newparticle_center)
    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    rotation_param = particles.rotation_param
    
    MD_var = step_fn(all_variables)

    New_particles = pc.particle(Newparticle_center,param_geometry,displacement_param,rotation_param,mygrids,shape_fn,Displacement_EQ,particles.Rotation_EQ)
    
    return pc.All_Variables(New_particles,velocity,pressure,Drag,Step_count,MD_var)
    
    
def Update_particle_position_Multiple(all_variables,dt):
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t =velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param
    New_eq = lambda t:Displacement_EQ(displacement_param,t)
    dx_dt = jax.jacrev(New_eq)

    
    
    U0 =dx_dt(current_t)
    #print(U0)
    Newparticle_center = jnp.array([particle_centers[:,0]+dt*U0[0],particle_centers[:,1]+dt*U0[1]]).T
    #print(Newparticle_center)
    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    rotation_param = particles.rotation_param
    
    MD_var = all_variables.MD_var

    New_particles = pc.particle(Newparticle_center,param_geometry,displacement_param,rotation_param,mygrids,shape_fn,Displacement_EQ,particles.Rotation_EQ)
    
    return pc.All_Variables(New_particles,velocity,pressure,Drag,Step_count,MD_var)


def Update_particle_position_Deformable(all_variables, dt):
    particles = all_variables.particles
    velocity = all_variables.velocity
    Drag = all_variables.Drag
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    MD_var = all_variables.MD_var

    new_particles = []

    # Check if particles is iterable or single particle
    if isinstance(particles, (list, tuple)):
        iterator = particles
    else:
        iterator = [particles]
    for p in iterator:
        marker_positions = p.get_marker_positions()  # (n, 2)
        vx_interp = jnp.array([point_interpolation(pos, velocity[0]) for pos in marker_positions])
        vy_interp = jnp.array([point_interpolation(pos, velocity[1]) for pos in marker_positions])
        marker_velocities = jnp.stack([vx_interp, vy_interp], axis=1)

        new_marker_positions = marker_positions + dt * marker_velocities
        new_center = jnp.mean(new_marker_positions, axis=0)

        new_p = pc.particle(
            particle_center=new_center,
            geometry_param=p.geometry_param,
            displacement_param=p.displacement_param,
            rotation_param=p.rotation_param,
            Grid=p.Grid,
            shape=p.shape,
            Displacement_EQ=p.Displacement_EQ,
            Rotation_EQ=p.Rotation_EQ,
            marker_positions=new_marker_positions,
        )
        new_particles.append(new_p)

    # If only one particle, return single, else list
    if len(new_particles) == 1:
        new_particles = new_particles[0]

    return pc.All_Variables(new_particles, velocity, pressure, Drag, Step_count, MD_var)


