import jax.numpy as jnp
from jax import grad, jacrev, jacfwd, hessian, lax
from optimal_control_problem import OCP, OCPiterates
from jax import debug


def bwd_pass(ocp: OCP, iterates: OCPiterates, reg_param: float):
    def bwd_step(carry, inp):
        Vx, Vxx = carry
        x, u = inp

        lx = grad(ocp.stage_cost, 0)(x, u, ocp.goal_state)
        lu = grad(ocp.stage_cost, 1)(x, u, ocp.goal_state)
        fx = jacrev(ocp.dynamics, 0)(x, u)
        fu = jacrev(ocp.dynamics, 1)(x, u)
        lxx = hessian(ocp.stage_cost, 0)(x, u, ocp.goal_state)
        luu = hessian(ocp.stage_cost, 1)(x, u, ocp.goal_state)
        lux = jacfwd(jacrev(ocp.stage_cost, 1), 0)(x, u, ocp.goal_state)

        Quu = luu + fu.T @ Vxx @ fu
        Quu += reg_param * jnp.eye(Quu.shape[0])
        Quu = (Quu + Quu.T) / 2
        feasible = jnp.all(jnp.linalg.eigvals(Quu))
        Qx = lx + fx.T @ Vx
        Qu = lu + fu.T @ Vx
        Qxx = lxx + fx.T @ Vxx @ fx
        Qux = lux + fu.T @ Vxx @ fx

        K = -jnp.linalg.solve(Quu, Qux)
        k = -jnp.linalg.solve(Quu, Qu)

        Vx = Qx - K.T @ Quu @ k
        Vxx = Qxx - K.T @ Quu @ K
        dV = k.T @ Qu + 0.5 * k.T @ Quu @ k

        return (Vx, Vxx), (K, k, dV, feasible)

    xN = iterates.states[-1]
    VxN = grad(ocp.final_cost, 0)(xN, ocp.goal_state)
    VxxN = hessian(ocp.final_cost, 0)(xN, ocp.goal_state)

    _, bwd_pass_out = lax.scan(
        bwd_step, (VxN, VxxN), (iterates.states[:-1], iterates.controls), reverse=True
    )
    gain, ffgain, diff_cost, bp_feasible = bwd_pass_out
    return gain, ffgain, jnp.sum(diff_cost), jnp.all(bp_feasible)


def fwd_pass(ocp: OCP, iterates: OCPiterates, gain: jnp.ndarray, ffgain: jnp.ndarray):
    def fwd_step(prev_x, inp):
        x, u, K, k = inp
        u = u + k + K @ (prev_x - x)
        x = ocp.dynamics(prev_x, u)
        return x, (x, u)

    _, fwd_pass_out = lax.scan(
        fwd_step,
        iterates.states[0],
        (iterates.states[:-1], iterates.controls, gain, ffgain),
    )
    new_states, new_controls = fwd_pass_out
    new_states = jnp.vstack((iterates.states[0], new_states))
    return new_states, new_controls


def ilqr(ocp: OCP, iterates0: OCPiterates):
    def ilqr_iteration(val):
        iterates, mu, nu = val

        cost = ocp.total_cost(iterates.states, iterates.controls, ocp.goal_state)
        debug.print("Cost:            {x}", x=cost)

        K, k, dV, bp_feasible = bwd_pass(ocp, iterates, mu)

        new_states, new_controls = fwd_pass(ocp, iterates, K, k)

        new_cost = ocp.total_cost(new_states, new_controls, ocp.goal_state)
        debug.print("New Cost:        {x}", x=new_cost)
        debug.print("bp_feasible:     {x}", x=bp_feasible)

        v_change = cost - new_cost

        gain_ratio = v_change / (-dV)

        def accept():
            return (
                mu * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                2.0,
                OCPiterates(new_states, new_controls),
            )

        def reject():
            return mu * nu, nu * 2.0, iterates

        mu, nu, iterates = lax.cond(
            jnp.logical_and(gain_ratio > 0, bp_feasible), accept, reject
        )
        debug.print("reg param:       {x}", x=mu)
        debug.print("------------------------")
        debug.breakpoint()
        return iterates, mu, nu

    def ilqr_conv(val):
        _, mu, _ = val
        tol = 1e-4
        exit_condition = mu < tol
        return jnp.logical_not(exit_condition)

    sol, _, _ = lax.while_loop(ilqr_conv, ilqr_iteration, (iterates0, 1.0, 2.0))
    return sol.states, sol.controls
