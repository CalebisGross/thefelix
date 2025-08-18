# Mathematical Model of the Felix Framework

**Document Version**: 1.0  
**Date**: 2025-08-18  
**Status**: Formal Specification  
**Implementation**: `src/core/helix_geometry.py`

## Abstract

This document provides the formal mathematical specification for the Felix Framework's helix-based multi-agent architecture. The model translates the 3D geometric visualization from `thefelix.md` into rigorous mathematical formulations suitable for theoretical analysis, implementation validation, and research publication.

## Core Mathematical Framework

### 1. Parametric Helix Definition

The Felix Framework helix is defined as a parametric curve in 3D space with time-dependent radius tapering.

#### 1.1 Basic Parameters

- **Height**: $H \in \mathbb{R}^+$ (total vertical extent)
- **Turns**: $n \in \mathbb{N}$ (complete rotations)
- **Top radius**: $R_{\text{top}} \in \mathbb{R}^+$ (radius at $t=1$)
- **Bottom radius**: $R_{\text{bottom}} \in \mathbb{R}^+$ (radius at $t=0$)
- **Parameter**: $t \in [0,1]$ (normalized path parameter)

**Constraint**: $R_{\text{top}} > R_{\text{bottom}} > 0$

#### 1.2 Parametric Equations

The helix position vector $\mathbf{r}(t)$ is defined as:

$$\mathbf{r}(t) = \begin{pmatrix} x(t) \\ y(t) \\ z(t) \end{pmatrix} = \begin{pmatrix} R(t) \cos(\theta(t)) \\ R(t) \sin(\theta(t)) \\ H \cdot t \end{pmatrix}$$

Where:
- **Height function**: $z(t) = H \cdot t$
- **Angular function**: $\theta(t) = 2\pi n t$  
- **Radius function**: $R(t) = R_{\text{bottom}} \left(\frac{R_{\text{top}}}{R_{\text{bottom}}}\right)^t$

### 2. Radius Tapering Function

#### 2.1 Exponential Tapering

The radius varies exponentially along the helix height:

$$R(t) = R_{\text{bottom}} \cdot \exp\left(t \ln\left(\frac{R_{\text{top}}}{R_{\text{bottom}}}\right)\right)$$

This can also be written as:
$$R(t) = R_{\text{bottom}} \left(\frac{R_{\text{top}}}{R_{\text{bottom}}}\right)^t$$

#### 2.2 Properties

- **Monotonicity**: $\frac{dR}{dt} = R(t) \ln\left(\frac{R_{\text{top}}}{R_{\text{bottom}}}\right) > 0$
- **Boundary conditions**: 
  - $R(0) = R_{\text{bottom}}$
  - $R(1) = R_{\text{top}}$
- **Smoothness**: $R(t) \in C^\infty([0,1])$

### 3. Geometric Properties

#### 3.1 Tangent Vector

The unit tangent vector $\mathbf{T}(t)$ is:

$$\mathbf{T}(t) = \frac{\mathbf{r}'(t)}{|\mathbf{r}'(t)|}$$

Where the derivative is:
$$\mathbf{r}'(t) = \begin{pmatrix} 
R'(t)\cos(\theta(t)) - R(t)\theta'(t)\sin(\theta(t)) \\
R'(t)\sin(\theta(t)) + R(t)\theta'(t)\cos(\theta(t)) \\
H
\end{pmatrix}$$

With:
- $R'(t) = R(t) \ln\left(\frac{R_{\text{top}}}{R_{\text{bottom}}}\right)$
- $\theta'(t) = 2\pi n$

#### 3.2 Arc Length

The arc length element is:
$$ds = |\mathbf{r}'(t)| dt$$

Where:
$$|\mathbf{r}'(t)| = \sqrt{(R'(t))^2 + (R(t)\theta'(t))^2 + H^2}$$

The total arc length from $t_1$ to $t_2$ is:
$$L(t_1, t_2) = \int_{t_1}^{t_2} \sqrt{(R'(t))^2 + (R(t) \cdot 2\pi n)^2 + H^2} \, dt$$

#### 3.3 Curvature

The curvature $\kappa(t)$ is:
$$\kappa(t) = \frac{|\mathbf{r}'(t) \times \mathbf{r}''(t)|}{|\mathbf{r}'(t)|^3}$$

This measures how sharply the helix bends at parameter $t$.

#### 3.4 Torsion

The torsion $\tau(t)$ measures the helix's twist:
$$\tau(t) = \frac{(\mathbf{r}' \times \mathbf{r}'') \cdot \mathbf{r}'''}{|\mathbf{r}' \times \mathbf{r}''|^2}$$

## 4. Agent Distribution Functions

### 4.1 Agent Spawn Distribution

Agents spawn according to a uniform random distribution:
$$T_i \sim \mathcal{U}(0,1), \quad i = 1, 2, \ldots, N$$

Where $T_i$ is the spawn time for agent $i$, and $N$ is the total number of agents.

#### 4.2 Agent Density Function

The expected agent density at parameter $t$ and time $\tau$ is:

$$\rho(t, \tau) = \sum_{i=1}^N \mathbb{P}(T_i \leq \tau \text{ and } T_i + P_i \geq \tau) \cdot \delta(t - (T_i + (\tau - T_i)))$$

Where $P_i$ is the processing time for agent $i$.

For large $N$, this approaches:
$$\rho(t, \tau) \approx N \cdot \mathbb{P}(T \leq \tau \text{ and } T + P \geq \tau) \cdot f_T(t)$$

Where $f_T$ is the probability density function of spawn times.

### 4.3 Attention Focusing Mechanism

The tapering radius creates natural attention focusing. The "attention density" at parameter $t$ is inversely related to the available circumferential space:

$$A(t) = \frac{1}{2\pi R(t)} = \frac{1}{2\pi R_{\text{bottom}} \left(\frac{R_{\text{top}}}{R_{\text{bottom}}}\right)^t}$$

This shows that attention density increases exponentially as $t \to 1$ (toward the narrow end).

## 5. Spoke Communication Geometry

### 5.1 Spoke Definition

A spoke from agent at position $\mathbf{r}(t)$ to the central post is the line segment:
$$\mathbf{s}(t, \lambda) = (1-\lambda)\mathbf{c}(t) + \lambda\mathbf{r}(t), \quad \lambda \in [0,1]$$

Where $\mathbf{c}(t) = (0, 0, Ht)$ is the central axis point at height $Ht$.

### 5.2 Spoke Length

The length of spoke from agent at parameter $t$ is:
$$L_{\text{spoke}}(t) = |\mathbf{r}(t) - \mathbf{c}(t)| = R(t)$$

This shows that communication "distance" varies with the tapering radius.

### 5.3 Communication Complexity

For $N$ agents using spoke-based communication:
- **Total connections**: $N$ (each agent to central post)
- **Message complexity**: $O(N)$ (linear scaling)
- **Maximum communication distance**: $R_{\text{top}}$

## 6. Numerical Implementation Notes

### 6.1 Discretization

For computational implementation, the continuous parameter $t$ is discretized:
$$t_k = \frac{k}{K}, \quad k = 0, 1, \ldots, K$$

Where $K$ is the number of discrete steps.

### 6.2 Arc Length Approximation

The arc length integral is approximated using trapezoidal rule:
$$L \approx \sum_{k=0}^{K-1} \frac{|\mathbf{r}'(t_k)| + |\mathbf{r}'(t_{k+1})|}{2} \cdot \frac{1}{K}$$

### 6.3 Validation Properties

The implementation should satisfy:
1. **Boundary conditions**: $\mathbf{r}(0) = (R_{\text{bottom}}, 0, 0)$, $\mathbf{r}(1) = (R_{\text{top}}, 0, H)$
2. **Continuity**: $\mathbf{r}(t)$ is continuous and differentiable
3. **Monotonicity**: $z(t)$ and $R(t)$ are strictly increasing
4. **Periodicity**: $\theta(t + 1/n) = \theta(t) + 2\pi$

## 7. OpenSCAD Model Correspondence

### 7.1 Parameter Mapping

| OpenSCAD Variable | Mathematical Symbol | Type |
|-------------------|-------------------|------|
| `height` | $H$ | Real |
| `turns` | $n$ | Integer |
| `top_radius` | $R_{\text{top}}$ | Real |
| `bottom_radius` | $R_{\text{bottom}}$ | Real |
| `step` | $k$ (discrete) | Integer |
| `$t` (animation) | $\tau$ (time) | Real |

### 7.2 Function Correspondence

The OpenSCAD `get_position(step, p_turns, p_segs, p_h, p_t_rad, p_b_rad)` function corresponds to:
$$\mathbf{r}\left(\frac{\text{step}}{\text{total\_steps}}\right)$$

Where $\text{total\_steps} = \text{p\_turns} \times \text{p\_segs}$.

## 8. Theoretical Implications

### 8.1 Convergence Properties

As agents progress from $t=0$ to $t=1$:
- Available circumferential space decreases exponentially
- Agent density increases, promoting interaction
- Processing focus naturally narrows (attention mechanism)

### 8.2 Stability Analysis

The system exhibits:
- **Geometric stability**: Bounded trajectories within the helix volume
- **Communication stability**: Bounded spoke lengths $\leq R_{\text{top}}$
- **Processing stability**: Finite processing time bounds

### 8.3 Scalability Properties

The mathematical model supports:
- **Agent scalability**: $O(N)$ space and communication complexity
- **Geometric scalability**: Parameters can be adjusted for larger/smaller systems
- **Computational scalability**: All functions have polynomial evaluation complexity

## 9. Applications to Research Hypotheses

This mathematical framework provides the foundation for:

- **H1 (Task Distribution)**: Statistical analysis of agent workload variance using $\rho(t,\tau)$
- **H2 (Communication Efficiency)**: Complexity analysis showing $O(N)$ vs $O(N^2)$ scaling
- **H3 (Attention Focusing)**: Formal proof using attention density function $A(t)$

## References

1. OpenSCAD implementation: `thefelix.md`
2. Python implementation: `src/core/helix_geometry.py`
3. Validation script: `validate_openscad.py`
4. Test suite: `tests/unit/test_helix_geometry.py`

---

**Note**: This mathematical model provides the theoretical foundation for the Felix Framework implementation and serves as the reference specification for all numerical computations and theoretical analysis.