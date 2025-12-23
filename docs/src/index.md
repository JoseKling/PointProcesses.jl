# PointProcesses.jl

Welcome to the documentation of [PointProcesses.jl](https://github.com/JoseKling/PointProcesses.jl), a package for modeling, simulation and inference with temporal point processes.

!!! warning "PointProcesses.jl is under active development"
    While there is lots of functionality, certain elements may not be polished,
    and bugs likely exist. Please report issues on GitHub.

## What are Point Processes?

Briefly, point processes are statistical models of random events in time or space. Point processes are parameterized by their intensity functions, or the instantaneous rate in time. A such, these models are useful anytime one is interested in how often one would see an event (e.g., buses arriving, earthquakes occuring, neurons firing.)

## Installation

Install the latest release from the General registry:

```julia
import Pkg; Pkg.add("PointProcesses")
```

Or install the development version:

```julia
import Pkg; Pkg.add(url="https://github.com/JoseKling/PointProcesses.jl")
```

## Quick Start

### Simulate a Poisson Process

```julia
using PointProcesses

# Create a Poisson process with intensity λ = 5.0 events per unit time
pp = PoissonProcess(5.0)

# Simulate events in the interval [0, 10]
history = simulate(pp, 0.0, 10.0)

# Examine the events
event_times(history)  # Get event timestamps
nb_events(history)    # Count events
```

### Fit a Hawkes Process

```julia
using PointProcesses

# Fit a self-exciting Hawkes process to observed data
fitted_hawkes = fit(HawkesProcess, history)

# The fitted process has parameters:
# - μ (mu): baseline intensity
# - α (alpha): excitation magnitude
# - ω (omega): decay rate
```

### Model Time-Varying Intensity

```julia
using PointProcesses

# Create a process with sinusoidal intensity
intensity_func = SinusoidalIntensity(
    amplitude=10.0,
    baseline=5.0,
    frequency=2π,
    phase=0.0
)
ipp = InhomogeneousPoissonProcess(intensity_func)

# Simulate and fit to data
simulated = simulate(ipp, 0.0, 10.0)
fitted_ipp = fit(InhomogeneousPoissonProcess{PolynomialIntensity{2}}, simulated)
```

## Available Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `PoissonProcess` | Constant intensity (homogeneous) | λ (intensity) |
| `MultivariatePoissonProcess` | Multiple independent streams | λᵢ for each dimension |
| `InhomogeneousPoissonProcess` | Time-varying intensity λ(t) | Varies by intensity function |
| `HawkesProcess` | Self-exciting (clustering) | μ, α, ω |
| `MarkedPoissonProcess` | Events with associated marks | λ, mark distribution |

## Parametric Intensity Functions

For inhomogeneous Poisson processes, choose from:

- **`PolynomialIntensity`**: λ(t) = a₀ + a₁t + a₂t² + ...
- **`ExponentialIntensity`**: λ(t) = a·exp(b·t)
- **`SinusoidalIntensity`**: λ(t) = a + b·sin(ωt + φ)
- **`PiecewiseConstantIntensity`**: Step function (histogram)
- **`LinearCovariateIntensity`**: Log-linear with covariates

## Core Workflow

```julia
# 1. Create or fit a point process model
pp = PoissonProcess(3.0)

# 2. Simulate event sequences
history = simulate(pp, 0.0, 100.0)

# 3. Fit to observed data
fitted_pp = fit(PoissonProcess, observed_history)

# 4. Validate the model
test = BootstrapTest(KSDistance(Exponential), PoissonProcess, history; n_sims=1000)
pval = pvalue(test)  # p-value for goodness-of-fit
```

## Mathematical Background

Temporal point processes are stochastic models characterized by their **conditional intensity function** λ(t|ℋₜ), which represents the instantaneous rate of events at time t given the history ℋₜ of events up to that time.
.

- [rasmussen2018](@cite) -- Compact introductory notes on point processes
- [Laub2021](@cite) -- More detailed, but still introductory level, material focused on Hawkes processes
- [DaleyJones2013](@cite) -- Comprehensive and formal presentation of point processes 


## Citation

If you use PointProcesses.jl in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8157372.svg)](https://doi.org/10.5281/zenodo.8157372)
