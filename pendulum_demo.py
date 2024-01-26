import jax.numpy as jnp
from utils import discretize_dynamics, wrap_angle, simulate_dynamics
from jax import random, config, debug, vmap
from optimal_control_problem import OCP, OCPiterates
from ilqr_jax import ilqr
import matplotlib.pyplot as plt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU and mute all warnings if no GPU/TPU is found
config.update("jax_platform_name", "cpu")


def pendulum_ode(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    gravity = 9.81
    length = 1.0
    mass = 1.0
    damping = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            -gravity / length * jnp.sin(position)
            + (action - damping * velocity) / (mass * length**2),
        )
    )


simulation_step = 0.01
down_sampling = 5
dynamics = discretize_dynamics(pendulum_ode, simulation_step, down_sampling)


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e1, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    c = 0.5 * _wrapped.T @ final_state_cost @ _wrapped
    return c


def stage_cost(
    state: jnp.ndarray, action: jnp.ndarray, goal_state: jnp.ndarray
) -> float:
    state_cost = jnp.diag(jnp.array([1e1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    c = 0.5 * _wrapped.T @ state_cost @ _wrapped
    c += 0.5 * action.T @ action_cost @ action
    return c


def total_cost(states: jnp.ndarray, actions: jnp.ndarray, goal_state: jnp.ndarray):
    li = vmap(stage_cost, in_axes=(0, 0, None))(states[:-1], actions, goal_state)
    lf = final_cost(states[-1], goal_state)
    return lf + jnp.sum(li)


def constraints(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    action_ub = 5.0
    action_lb = 5.0
    return jnp.hstack((action - action_ub, -action + action_lb))


horizon = 40
mean = jnp.array([0.0])
sigma = jnp.array([0.1])
key = random.PRNGKey(465)
initial_controls = mean + sigma * random.normal(key, shape=(horizon, 1))
state0 = jnp.array([0.01, -0.01])
initial_states = simulate_dynamics(dynamics, state0, initial_controls)
desired_state = jnp.array((jnp.pi, 0.0))
# initial_lagrange = 0.01 * jnp.ones((horizon, 2))
# initial_slack = 0.1 * jnp.ones((horizon, 2))

pendulum_ocp = OCP(
    dynamics, constraints, stage_cost, final_cost, total_cost, desired_state
)
iterates = OCPiterates(initial_states, initial_controls)

opt_states, opt_controls = ilqr(pendulum_ocp, iterates)

plt.plot(opt_states[:, 0])
plt.show()
