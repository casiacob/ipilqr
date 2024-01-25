import jax.numpy as jnp
from utils import discretize_dynamics, wrap_angle
from jax import lax, random, config, debug
from optimal_control_problem import OCP, OCPiterates
from typing import Callable

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


def simulate_dynamics(
    dynamics: Callable, initial_state: jnp.ndarray, control_array: jnp.ndarray
) -> jnp.ndarray:
    def body_scan(prev_state, control):
        return dynamics(prev_state, control), dynamics(prev_state, control)

    _, states = lax.scan(body_scan, initial_state, control_array)
    states = jnp.vstack((initial_state, states))
    return states


def final_cost(state: jnp.ndarray, goal_state: jnp.ndarray) -> float:
    final_state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    c = 0.5 * _wrapped.T @ final_state_cost @ _wrapped
    return c


def stage_cost(
    state: jnp.ndarray, action: jnp.ndarray, goal_state: jnp.ndarray
) -> float:
    state_cost = jnp.diag(jnp.array([2e0, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    c = 0.5 * _wrapped.T @ state_cost @ _wrapped
    c += 0.5 * action.T @ action_cost @ action
    return c


def constraints(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    action_ub = 5.0
    action_lb = 5.0
    return jnp.hstack((action - action_ub, -action + action_lb))


horizon = 40
mean = jnp.array([0.0])
sigma = jnp.array([0.01])
key = random.PRNGKey(465)
initial_controls = mean + sigma * random.normal(key, shape=(horizon, 1))
state0 = jnp.array([wrap_angle(0.01), -0.01])
initial_states = simulate_dynamics(dynamics, state0, initial_controls)
desired_state = jnp.array((jnp.pi, 0.0))
initial_lagrange = 0.01 * jnp.ones((horizon, 2))
initial_slack = 0.1 * jnp.ones((horizon, 2))

pendulum = OCP(dynamics, constraints, stage_cost, final_cost, desired_state)
iterates = OCPiterates(
    initial_states, initial_controls, initial_lagrange, initial_slack
)
debug.breakpoint()