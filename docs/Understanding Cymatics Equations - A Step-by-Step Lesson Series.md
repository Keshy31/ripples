# Understanding Cymatics Equations: A Step-by-Step Lesson Series

I've structured this as a series of interconnected lessons on the equations from the cymatics simulator technical overview. Each lesson builds on the previous one, starting from basic wave concepts and progressing to advanced topics like numerical simulations. We'll use simple explanations, examples, and visuals to make it educational. By the end, you'll have a solid grasp of how these equations model vibrational patterns.

## Lesson 1: The Basics of Waves and the 2D Wave Equation

Let's start with the foundation: waves. Waves are disturbances that propagate through a medium, like ripples on water or sound through air. In cymatics, we're interested in how vibrations create visible patterns on surfaces.

The core equation here is the **2D wave equation**, which describes how displacement \(u(x, y, t)\) (e.g., the height of a vibrating plate at position (x,y) and time t) evolves:

\[
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u + f(x, y, t)
\]

- **Breakdown**:
  - Left side: \(\frac{\partial^2 u}{\partial t^2}\) is the acceleration of the displacement over time. It tells us how fast the surface is changing.
  - Right side: \(c^2 \nabla^2 u\) represents the spatial curvature. \(\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\) is the Laplacian operator, measuring how much the displacement differs from its neighbors (like tension pulling a string back).
  - \(c\) is the wave speed, depending on the medium (e.g., faster in stiffer materials).
  - \(f(x, y, t)\) is an optional forcing term, like a speaker inputting energy (e.g., a sine wave).

This is a partial differential equation (PDE) because it involves derivatives in multiple variables. It models how waves spread out in 2D space over time.

**Example**: Imagine dropping a pebble in a pond—the wave equation predicts the expanding ripples. In cymatics, boundaries (like plate edges) reflect waves, leading to interference.

**Why it matters**: This equation is universal for wave phenomena. Understanding it sets the stage for why patterns form when waves "stand still."

## Lesson 2: Standing Waves, Resonance, and the Helmholtz Equation

Building on Lesson 1, waves don't always propagate freely—in bounded systems like plates, they reflect and interfere, creating **standing waves**. These are stationary patterns where certain points (nodes) don't move, and others (antinodes) oscillate maximally.

To analyze standing waves, we assume a harmonic (sine-like) time dependence: \(u(x,y,t) = \phi(x,y) e^{i\omega t}\), where \(\omega = 2\pi f\) is angular frequency. Plugging this into the wave equation (ignoring \(f\) for free vibrations) simplifies it to the **Helmholtz equation**:

\[
\nabla^2 \phi + k^2 \phi = 0, \quad k = \frac{\omega}{c}
\]

- **Breakdown**:
  - This is time-independent, focusing on spatial modes \(\phi(x,y)\).
  - It's an eigenvalue problem: Solutions exist only for specific \(k\) (wavenumbers), corresponding to resonant frequencies where energy builds up.
  - Nodes are where \(\phi = 0\); in cymatics, sand gathers here.

**Resonance**: When you drive the system at these frequencies (via \(f\)), amplitudes grow, forming patterns.

**Example**: Think of a guitar string—plucking it excites modes at harmonics. In 2D, it's like a drumhead.

This lesson transitions us from dynamic waves to static patterns, preparing for specific shapes in plates.

## Lesson 3: Chladni Modes for Square Plates

Now that we have the Helmholtz equation, let's solve it for a square plate (side length \(L\)), with fixed boundaries (\(\phi=0\) at edges). The solutions are **Chladni modes**:

\[
\phi_{n,m}(x,y) = \sin\left(\frac{n\pi x}{L}\right) \sin\left(\frac{m\pi y}{L}\right)
\]

Eigenfrequencies:

\[
f_{n,m} = \frac{c}{2L} \sqrt{n^2 + m^2}
\]

- **Breakdown**:
  - \(n, m\) are positive integers (mode numbers), determining nodal lines: \(n\) horizontal, \(m\) vertical.
  - Sine functions satisfy boundaries (zero at x=0,L and y=0,L).
  - Frequency increases with higher modes, creating more complex grids or crosses.

**How it builds on previous**: The Helmholtz equation is solved using separation of variables: Assume \(\phi = X(x)Y(y)\), leading to sine solutions from 1D string analogies.

**Example**: For (n=1,m=1), it's a simple hump; for (2,3), wavy lines. These predict sand patterns on vibrating squares.








These images show real and simulated square Chladni patterns, illustrating how nodes form lines.

## Lesson 4: Chladni Modes for Circular Plates

Extending square plates to circles (radius \(R\)) requires polar coordinates (r, θ). The Helmholtz solutions involve **Bessel functions**:

\[
\phi_{n,s}(r,\theta) = J_n(k_{n,s} r / R) \cos(n\theta)
\]

Frequencies:

\[
f_{n,s} = \frac{c k_{n,s}}{2\pi R}
\]

- **Breakdown**:
  - \(J_n\) is the nth-order Bessel function, oscillating like damped sines.
  - \(k_{n,s}\) is the sth root where \(J_n(k)=0\) (boundary condition: zero at r=R).
  - \(n\): Azimuthal modes (angular lines), \(s\): Radial modes (concentric circles).
  - Cosine gives rotational symmetry.

**Connection to prior lessons**: Separation in polar: Radial part yields Bessel (from Helmholtz in cylindrical coords), angular is trig functions.

**Example**: For (n=0,s=1), a bullseye; higher n adds spokes. Ideal for bowl-like cymatics.








Visuals here depict circular patterns, showing Bessel's radial decay.

## Lesson 5: Faraday Waves in Vibrating Fluids

Shifting to fluids (like water in a bowl), gravity and tension modify the wave equation. For height perturbation \(h(x,y,t)\):

\[
\frac{\partial^2 h}{\partial t^2} + g \nabla^2 h - \frac{\sigma}{\rho} \nabla^4 h = 0
\]

- **Breakdown**:
  - Gravity term \(g \nabla^2 h\): Restores flatness for large waves.
  - Surface tension \(\frac{\sigma}{\rho} \nabla^4 h\): Bi-Laplacian smooths small ripples.
  - For vertical driving (Faraday setup), parametric instability creates subharmonics (patterns at half frequency).

**Building on earlier**: Starts from wave equation but adds fluid physics—nonlinear terms (not shown) cause hexagons or chaos at high amplitudes.

**Example**: Shaking a fluid vertically forms lattices, unlike plate nodes.








These show evolving Faraday patterns, highlighting symmetry.

## Lesson 6: Numerical Methods for Simulating Waves

Analytical solutions (Lessons 3-5) are ideal, but for complex or driven systems, we discretize. Using finite differences on the wave equation:

\[
u^{t+1}_{i,j} = 2u^t_{i,j} - u^{t-1}_{i,j} + c^2 \Delta t^2 (\nabla^2 u^t)_{i,j}
\]

Discrete Laplacian:

\[
(\nabla^2 u)_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{\Delta x^2}
\]

- **Breakdown**:
  - Time-step: Second-order central difference approximates \(\partial^2 u / \partial t^2\).
  - Space: 5-point stencil for Laplacian.
  - Stability: \(\frac{c \Delta t}{\Delta x} \leq 1\) (CFL condition).

**Ties back**: Approximates the continuous PDE for computation, allowing simulation of any setup.

**Example**: In code, this creates animations of ripples evolving into patterns.








These illustrate numerical wave simulations, showing discrete evolution.

Congratulations! You've progressed from basic waves to full simulations. Practice by tweaking parameters in thought experiments or code. If you have questions, ask!