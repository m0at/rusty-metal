"""Simulation & Physics: heat equation, wave equation, stencils, RK4, Verlet, Monte Carlo."""

import numpy as np
from bench.harness import bench_fn, HAS_TORCH, HAS_TORCH_MPS, HAS_MLX, _sync_mps, _sync_mlx

if HAS_TORCH:
    import torch
if HAS_MLX:
    import mlx.core as mx


def run(n: int = 10_000_000):
    grid = 1024
    total = grid * grid

    # ── Heat equation (one step, 5-point stencil) ──
    u_np = np.random.randn(grid, grid).astype(np.float32)
    alpha, dt, dx = 0.1, 0.0001, 0.01
    r = alpha * dt / (dx * dx)

    bench_fn("sim_heat_equation", "simulation", "numpy",
             lambda: _np_heat_step(u_np, r), total)
    if HAS_TORCH:
        u_cpu = torch.from_numpy(u_np)
        bench_fn("sim_heat_equation", "simulation", "torch_cpu",
                 lambda: _torch_heat_step(u_cpu, r), total)
        if HAS_TORCH_MPS:
            u_mps = u_cpu.to("mps")
            bench_fn("sim_heat_equation", "simulation", "torch_mps",
                     lambda: _torch_heat_step(u_mps, r), total, sync=_sync_mps)

    # ── Wave equation (one step, leapfrog) ──
    u_prev = np.random.randn(grid, grid).astype(np.float32)
    u_curr = np.random.randn(grid, grid).astype(np.float32)
    c = 1.0
    courant = (c * dt / dx) ** 2

    bench_fn("sim_wave_equation", "simulation", "numpy",
             lambda: _np_wave_step(u_prev, u_curr, courant), total)
    if HAS_TORCH:
        up_cpu = torch.from_numpy(u_prev)
        uc_cpu = torch.from_numpy(u_curr)
        bench_fn("sim_wave_equation", "simulation", "torch_cpu",
                 lambda: _torch_wave_step(up_cpu, uc_cpu, courant), total)

    # ── 2D stencil (general 5-point) ──
    bench_fn("sim_stencil_2d", "simulation", "numpy",
             lambda: _np_stencil_2d(u_np), total)
    if HAS_TORCH:
        bench_fn("sim_stencil_2d", "simulation", "torch_cpu",
                 lambda: _torch_stencil_2d(u_cpu), total)

    # ── 3D stencil ──
    grid3d = 128
    total3d = grid3d ** 3
    u3d = np.random.randn(grid3d, grid3d, grid3d).astype(np.float32)
    bench_fn("sim_stencil_3d", "simulation", "numpy",
             lambda: _np_stencil_3d(u3d), total3d)

    # ── RK4 integration (N independent ODEs: dy/dt = -y) ──
    n_odes = 100_000
    y_np = np.random.randn(n_odes).astype(np.float32)
    dt_rk4 = 0.01

    bench_fn("integrate_rk4", "simulation", "numpy",
             lambda: _np_rk4_step(y_np, dt_rk4), n_odes)
    if HAS_TORCH:
        y_cpu = torch.from_numpy(y_np)
        bench_fn("integrate_rk4", "simulation", "torch_cpu",
                 lambda: _torch_rk4_step(y_cpu, dt_rk4), n_odes)
        if HAS_TORCH_MPS:
            y_mps = y_cpu.to("mps")
            bench_fn("integrate_rk4", "simulation", "torch_mps",
                     lambda: _torch_rk4_step(y_mps, dt_rk4), n_odes, sync=_sync_mps)

    # ── Velocity Verlet (N-body step, simplified) ──
    n_particles = 10_000
    pos = np.random.randn(n_particles, 3).astype(np.float32)
    vel = np.random.randn(n_particles, 3).astype(np.float32)
    acc = np.random.randn(n_particles, 3).astype(np.float32)

    bench_fn("integrate_verlet", "simulation", "numpy",
             lambda: _np_verlet_step(pos, vel, acc, 0.001), n_particles * 3)
    if HAS_TORCH:
        pos_cpu = torch.from_numpy(pos)
        vel_cpu = torch.from_numpy(vel)
        acc_cpu = torch.from_numpy(acc)
        bench_fn("integrate_verlet", "simulation", "torch_cpu",
                 lambda: _torch_verlet_step(pos_cpu, vel_cpu, acc_cpu, 0.001), n_particles * 3)

    # ── Monte Carlo integration (estimate pi) ──
    mc_n = 1_000_000
    bench_fn("monte_carlo_integrate", "simulation", "numpy",
             lambda: _np_monte_carlo_pi(mc_n), mc_n)


# ── Numpy implementations ──

def _np_heat_step(u, r):
    laplacian = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)
    return u + r * laplacian

def _np_wave_step(u_prev, u_curr, courant):
    laplacian = (np.roll(u_curr, 1, 0) + np.roll(u_curr, -1, 0) + np.roll(u_curr, 1, 1) + np.roll(u_curr, -1, 1) - 4 * u_curr)
    return 2 * u_curr - u_prev + courant * laplacian

def _np_stencil_2d(u):
    return 0.25 * (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1))

def _np_stencil_3d(u):
    return (1.0/6.0) * (np.roll(u,1,0) + np.roll(u,-1,0) + np.roll(u,1,1) + np.roll(u,-1,1) + np.roll(u,1,2) + np.roll(u,-1,2))

def _np_rk4_step(y, dt):
    f = lambda y: -y  # simple decay
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _np_verlet_step(pos, vel, acc, dt):
    new_pos = pos + vel * dt + 0.5 * acc * dt * dt
    new_acc = -new_pos * 0.01  # simple spring force
    new_vel = vel + 0.5 * (acc + new_acc) * dt
    return new_pos, new_vel

def _np_monte_carlo_pi(n):
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    return 4.0 * np.mean(x*x + y*y <= 1.0)


if HAS_TORCH:
    def _torch_heat_step(u, r):
        laplacian = (torch.roll(u,1,0) + torch.roll(u,-1,0) + torch.roll(u,1,1) + torch.roll(u,-1,1) - 4*u)
        return u + r * laplacian

    def _torch_wave_step(u_prev, u_curr, courant):
        laplacian = (torch.roll(u_curr,1,0) + torch.roll(u_curr,-1,0) + torch.roll(u_curr,1,1) + torch.roll(u_curr,-1,1) - 4*u_curr)
        return 2*u_curr - u_prev + courant * laplacian

    def _torch_stencil_2d(u):
        return 0.25 * (torch.roll(u,1,0) + torch.roll(u,-1,0) + torch.roll(u,1,1) + torch.roll(u,-1,1))

    def _torch_rk4_step(y, dt):
        f = lambda y: -y
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _torch_verlet_step(pos, vel, acc, dt):
        new_pos = pos + vel * dt + 0.5 * acc * dt * dt
        new_acc = -new_pos * 0.01
        new_vel = vel + 0.5 * (acc + new_acc) * dt
        return new_pos, new_vel
