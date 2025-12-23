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
with the simplest point process model and making the connection between
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
- **Inhomogeneous Poisson process**: λ(t) varies with time.
- **History-dependent processes**: λ(t | 𝓗ₜ) depends on past events (e.g. Hawkes processes)

### The Homogeneous Poisson Process

The simplest and most widely used point process is the **homogeneous Poisson process**.
It is defined by a single parameter λ > 0, the constant event rate, and is characterized by:

1. **Independent increments**: The numbers of events in disjoint time intervals are independent.
2. **Stationary increments**: The distribution of events depends only on the length of the interval.
3. **Orderliness**: The probability of more than one event occurring in a small interval is negligible.

Despite its simplicity, the homogeneous Poisson process provides a useful baseline model
and a reference point for more expressive point process models. Let's start simulating data using PointProcesses.jl

````@example Basics
using PointProcesses
using Distributions
using HypothesisTests
using StatsBase
using Plots

using StableRNGs # hide
rng = StableRNG(67); # hide
nothing #hide
````

Usually we observe some data, typically, these will be event times. Assume, we have seen the following event times, from a cultured retinal ganglion cell neuron with no input stimulus:

````@example Basics
waiting_times = rand(rng, InverseGaussian(0.2, 0.5), 50) # hide
event_times = cumsum(waiting_times) # hide

h = History(event_times, 0.0, maximum(event_times) + 1.0)
````

Let's visualize our event times as a raster plot:

````@example Basics
raster = plot(;
    xlabel="Time",
    ylabel="Events",
    title="Raster Plot of Event Times",
    ylim=(-0.1, 1),
    yticks=false,
)
for time in h.times
    vline!([time]; color=:blue, linewidth=2, label=false)
end

raster
````

Looking at this plot its difficult to tell whether the event times are consistent with a homogeneous Poisson process.
We can bin the data in small time windows to get a sense of the event rate over time. From the assumptiosn above, we expect the distirbution of
counts to be roughly Poisson distributed with rate λ * Δt in each bin of width Δt.

````@example Basics
bin_width = 1.0
bins = collect(h.tmin:bin_width:h.tmax)           # bin edges
counts = fit(Histogram, h.times, bins).weights;    # counts per bin
nothing #hide
````

Plot counts over time (counts/bin_width is a crude rate estimate)

````@example Basics
bin_centers = (bins[1:(end - 1)] .+ bins[2:end]) ./ 2

p_counts = plot(
    bin_centers,
    counts;
    seriestype=:bar,
    xlabel="Time",
    ylabel="Count per bin",
    title="Binned event counts",
)

p_counts
````

This is still difficult to tell. To get a better sense of whether the data is consistent with a homogeneous Poisson process, we can consider an alternative view of the homogeenous
point process based on the distribution of its waiting times. We discuss this below.

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

Lets now calcualte the waiting times and see how "exponential" they look:

````@example Basics
waiting_times = diff(vcat(h.tmin, h.times));   # includes waiting time from tmin to first event
nothing #hide
````

````@example Basics
T = duration(h)             # should be h.tmax - h.tmin
n = length(h)               # number of events
λ_est = n / T

p_waiting = histogram(
    waiting_times;
    bins=10,
    normalize=:pdf,
    xlabel="Waiting time",
    ylabel="Density",
    title="Histogram of Waiting Times",
    label="Empirical",
)

x = range(0; stop=maximum(waiting_times), length=400)
plot!(
    p_waiting,
    x,
    pdf.(Exponential(1 / λ_est), x);
    label="Exponential(λ̂=$(round(λ_est, digits=2)))",
    linewidth=2,
    color=:red,
)

p_waiting
````

This could be maybe exponential, but there is some signs that it is not. For example, why do we not see any waiting times below approxiamtely 0.1s?
Let's fit the actual model now using PointProcesses.jl and run soem, formal statistical tests.

### Fitting a Homogeneous Poisson Process

````@example Basics
pp_model = fit(PoissonProcess{Float64,Dirac{Nothing}}, h)

println("Estimated rate λ̂ = ", pp_model.λ) # hide
````

We can now use this fitted model to run some statistical tests to see whether the homogeneous Poisson process is a good fit to the data. First let's plot a qq-plot
of the waiting times against the expected exponential distribution.

````@example Basics
q = range(0.01, 0.99; length=200)
emp_q = quantile(waiting_times, q)
theo_q = quantile.(Exponential(1 / λ_est), q)   # Exp with mean 1/λ_est

pqq = plot(
    theo_q,
    emp_q;
    seriestype=:scatter,
    xlabel="Theoretical quantiles: Exp(mean=1/λ̂)",
    ylabel="Empirical waiting-time quantiles",
    title="QQ plot: waiting times vs fitted exponential",
    aspect_ratio=:equal,
    label=false,
    xlim=(0, 1),
    ylim=(0, 1),
)
plot!(pqq, theo_q, theo_q; label=false)  # y=x line
pqq
````

The QQ-plot shows some deviations from the expected exponential distribution, particularly at the lower quantiles. We can use time-rescaling
to formally test the goodness-of-fit of the homogeneous Poisson process model. The time-rescaling theorem states that if we transform the event times tᵢ using the integrated
intensity function, the transformed times should be distributed as a homogeneous Poisson process with unit rate. Luckily PointProcesses.jl provides a convenient function to do this transformation for us:

````@example Basics
transformed_times = time_change(h, pp_model)
transformed_waiting_times = diff(vcat(transformed_times.tmin, transformed_times.times))
````

Under the null hypothesis that the data is generated by the fitted homogeneous Poisson process, these transformed waiting times should be i.i.d. Exp(1) distributed.

````@example Basics
ks_test = ExactOneSampleKSTest(transformed_waiting_times, Exponential(1.0))
println("KS test p-value: ", pvalue(ks_test)) # hide
````

The KS test p-value is very small, indicating that we can reject the null hypothesis that the data was generated by a homogeneous Poisson process at conventional significance levels.
This suggests that the homogeneous Poisson process is not an adequate model for the observed event times. There are several possible reasons for this, including:
- The event rate may not be constant over time (inhomogeneous Poisson process)
- There may be history dependence (e.g., refractory periods, bursting)
But this is a topic for another tutorial!

