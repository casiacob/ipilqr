from typing import NamedTuple, Callable
import jax.numpy as jnp


class OCP(NamedTuple):
    dynamics: Callable
    constraints: Callable
    stage_cost: Callable
    final_cost: Callable
    total_cost: Callable
    goal_state: jnp.ndarray


class OCPiterates(NamedTuple):
    states: jnp.ndarray
    controls: jnp.ndarray
