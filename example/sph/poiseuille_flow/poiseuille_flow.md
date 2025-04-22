<!-- @ author: bcynuaa <bcynuaa@163.com>
@ date: 2025/04/22 17:31:05
@ license: MIT
@ language: Julia
@ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
@ description: -->

[toc]

# Poiseuille Flow

Poiseuille flow is a type of flow in a pipe where the flow is driven by a pressure gradient. The flow is laminar and the velocity profile is parabolic.

The governing equation for Poiseuille flow is the incomressible Navier-Stokes equation.

Usually, the pressure gradient can be performed by a body force along the pipe. Assuming the flow is the same along $x$ direction ($\partial/\partial x = 0$), the steady-state ($\partial/\partial t = 0$), and vertical velocity $v$ is zero ($v = 0$), the governing equation can be simplified to:

$$
\begin{equation}
    \mu \frac{\mathrm{d}^2u}{\mathrm{d}y^2}=\frac{\mathrm{d}p}{\mathrm{d}x}
\end{equation}
$$

Denote $f_x = -\frac{\mathrm{d}p}{\mathrm{d}x}$, the solution of the equation is:

$$
\begin{equation}
    u = -\frac{f_x}{2\mu}y^2 + C_1y + C_2
\end{equation}
$$

where $C_1$ and $C_2$ are constants. Apply the boundary condition $u(0) = 0$ and $u(h) = 0$, the solution is:

$$
\begin{equation}
    u = \frac{f_x}{2\mu}y(h-y)
\end{equation}
$$

The $\max(u)$ is at $y = h/2$, with value:

$$
\begin{equation}
    u_{\max} = \frac{f_x}{8\mu}h^2
\end{equation}
$$

And the mean velocity is:

$$
\begin{equation}
    u_{\text{mean}} = \frac{1}{h}
    \int_0^h u \mathrm{d}y = \frac{f_x}{12\mu}h^2
\end{equation}
$$

Also, when devided by density, the equation can be written as:

$$
\begin{equation}
    u(y) = \frac{a_x}{2\nu}y(h-y)
\end{equation}
$$

where $\nu = \mu/\rho$ is the kinematic viscosity, and $a_x = f_x/\rho$ is the acceleration.

When the flow is driven by a body force, starting from steady-state, a time-related series analytical solution can be obtained:

$$
\begin{equation}
    \begin{aligned}
        u(y, t) = &\frac{a_x}{2\nu}y(h-y) - \\
    &\sum_{n=0}^{\infty} \frac{4a_x h^2}{\nu \pi^3 (2n+1)^3}
    \sin\left[
        (2n + 1)\frac{\pi y}{h}
    \right]
    \exp\left[
        -\frac{\nu \pi^2 (2n+1)^2 t}{h^2}
    \right]
    \end{aligned}
\end{equation}
$$