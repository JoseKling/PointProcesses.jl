```@meta
EditURL = "../../examples/Basics.jl"
```

## The basics of Point Process Modeling

This tutorial provides a brief introduction to point process modeling using PointProcesses.jl.
We will cover the basic math of point processes, how to define basic models, simulate data,
and fit models to observed data. For further, and more advanced pedagogy, please refer to
any of the following great resources:

- Daley & Vere-Jones, *An Introduction to the Theory of Point Processes*, Volume I & II
- Cox & Isham, *Point Processes*
- (Add more resources here)

The focus of this tutorial is intentionally narrow: we aim to build intuition by starting
with the simplest nontrivial point process model and making the connection between
mathematical definitions, simulation, and likelihood-based inference explicit.

### Basic Point Process Concepts

A point process is a mathematical model for random events occurring in time or space.
In this tutorial, we focus on **temporal point processes**, where events are represented
as a set of event times:

    {t₁, t₂, …, tₙ} ⊂ [0, T]

The key components of a point process are:

- **Events**: The occurrences we are modeling, represented as points in time.
- **Counting Process**: N(t), the number of events that have occurred up to time t.
- **Intensity Function**: A (possibly time- and history-dependent) function λ(t | 𝓗ₜ)
  describing the instantaneous rate at which events occur.
- **History**: 𝓗ₜ, the record of events strictly before time t.

Intuitively, the intensity function λ(t | 𝓗ₜ) satisfies

    λ(t | 𝓗ₜ) Δt ≈ P(event in [t, t + Δt) | 𝓗ₜ)

for small Δt. Different choices of λ give rise to different classes of point processes:

- **Homogeneous Poisson process**: λ(t) = λ (constant)
- **Inhomogeneous Poisson process**: λ(t) varies deterministically with time
- **History-dependent processes**: λ(t | 𝓗ₜ) depends on past events (e.g. Hawkes processes)

### The Homogeneous Poisson Process

The simplest and most widely used point process is the **homogeneous Poisson process**.
It is defined by a single parameter λ > 0, the constant event rate, and is characterized by:

1. **Independent increments**: The numbers of events in disjoint time intervals are independent.
2. **Stationary increments**: The distribution of events depends only on the length of the interval.

Despite its simplicity, the homogeneous Poisson process provides a useful baseline model
and a reference point for more expressive point process models. Let's start simulating data using PointProcesses.jl

````@example Basics
using PointProcesses
using StableRNGs
using Plots
````

### Two Equivalent Views

The homogeneous Poisson process admits two equivalent and complementary descriptions:

**(1) Inter-event times**
The waiting times between successive events are independent and identically distributed:

    τₖ = tₖ - tₖ₋₁ ∼ Exponential(λ)

This view is particularly convenient for simulation.

**(2) Counting process**
The total number of events in a time window of length T is distributed as:

    N(T) ∼ Poisson(λ T)

Conditioned on N(T) = n, the event times are uniformly distributed on [0, T].

These two perspectives define the same stochastic process and will both be useful
for understanding likelihood-based inference.

