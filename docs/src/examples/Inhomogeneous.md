```@meta
EditURL = "../../examples/Inhomogeneous.jl"
```

# Fitting Data to an Inhomogeneous Poisson Process Model

This tutorial demonstrates how to fit data to an inhomogeneous point process model, as well as how to develop a new model from scratch.
We will generate point process data from a simulated "Hippocampal Place Cell" and then fit the model to the data.

## A Quick Recap of Inhomogeneous Poisson Processes
An inhomogeneous Poisson process is a type of point process where the intensity function varies over time or space.
The intensity function, denoted as ``λ(t)`` for time or ``λ(x)`` for space, describes the expected number of events per unit time or space at a given point.
In the context of neural data, inhomogeneous Poisson processes are often used to model the firing rates of neurons that change in response to stimuli or other factors.
Importantly, all Inhomogeneous Poisson processes have the same general likelihood function form, which is given by:
``L(λ; {t_i}) = \text{exp}(-∫ λ(t) dt) * ∏ λ(t_i)``
where ``{t_i}`` are the observed event times.
This likelihood function consists of two main components:
1. The exponential term ``\text{exp}(-∫ λ(t) dt)`` represents the probability of observing no events in the intervals where no events were recorded.
2. The product term ``∏ λ(t_i)`` accounts for the likelihood of observing events at the specific times ``{t_i}``.
This general form allows for flexibility in modeling various types of inhomogeneous Poisson processes by specifying different intensity functions ``λ(t)``.

````@example Inhomogeneous
using PointProcesses
using Distributions
using StableRNGs
using Plots
using StatsAPI
using SpecialFunctions: erf
````

## Simulating Hippocampal Place Cell Data
Hippocampal place cells are neurons that fire when an animal is in a specific location.
As the animal moves through space, the firing rate varies, creating an inhomogeneous Poisson process.
We'll simulate data where the animal makes multiple passes through the place field.

Let's create a Gaussian-like place field with peak firing at the center:

````@example Inhomogeneous
function gaussian_place_field(t; peak_rate=20.0, center=5.0, width=1.5)
    return peak_rate * exp(-((t - center)^2) / (2 * width^2))
end
````

Visualize the true intensity function:

````@example Inhomogeneous
t_range = 0.0:0.01:10.0
plot(
    t_range,
    gaussian_place_field.(t_range);
    xlabel="Position (arbitrary units)",
    ylabel="Firing rate (Hz)",
    label="True place field",
    linewidth=2,
    title="Hippocampal Place Cell Firing Rate",
)
````

Now let's simulate spike times from this intensity function:

````@example Inhomogeneous
rng = StableRNG(12345)
pp_true = InhomogeneousPoissonProcess(gaussian_place_field)
h = simulate(rng, pp_true, 0.0, 10.0)
````

Create a history object for our observed data:

````@example Inhomogeneous
println("Number of spikes observed: ", length(h))
````

Visualize the spike times as a raster plot:

````@example Inhomogeneous
scatter!(
    h.times,
    zeros(length(h));
    marker=:vline,
    markersize=10,
    label="Observed spikes",
    color=:red,
    alpha=0.6,
)
````

## Fitting Parametric Models
Now we'll try to recover the underlying intensity function by fitting different (non-)parametric models.

### 1. Piecewise Constant Intensity (Histogram Estimator)
The simplest approach is to bin the data and estimate a constant rate in each bin:

````@example Inhomogeneous
pp_piecewise = fit(
    InhomogeneousPoissonProcess{PiecewiseConstantIntensity{Float64},Dirac{Nothing}},
    h,
    20,  # number of bins
)
````

Visualize the fit:

````@example Inhomogeneous
plot(
    t_range,
    gaussian_place_field.(t_range);
    xlabel="Position",
    ylabel="Firing rate (Hz)",
    label="True intensity",
    linewidth=2,
    title="Piecewise Constant Fit",
)
plot!(
    t_range,
    pp_piecewise.intensity_function.(t_range);
    label="Fitted intensity",
    linewidth=2,
)
scatter!(
    h.times,
    zeros(length(h.times));
    marker=:vline,
    markersize=10,
    label="Spikes",
    color=:red,
    alpha=0.6,
)

### 2. Polynomial Intensity with Log Link
````

A polynomial model can capture smooth variations. We'll use a quadratic polynomial with log link
to ensure the intensity stays positive:

Initial parameter guess for a quadratic: ``\log(λ(t)) = a₀ + a₁*t + a₂*t²``

````@example Inhomogeneous
init_params = [2.0, 0.0, -0.1]

pp_poly = fit(PolynomialIntensity{Float64}, h, init_params; link=:log)
````

Visualize the polynomial fit:

````@example Inhomogeneous
plot(
    t_range,
    gaussian_place_field.(t_range);
    xlabel="Position",
    ylabel="Firing rate (Hz)",
    label="True intensity",
    linewidth=2,
    title="Polynomial Intensity Fit (Log Link)",
)
plot!(t_range, pp_poly.(t_range); label="Fitted intensity", linewidth=2)
scatter!(
    h.times,
    zeros(length(h.times));
    marker=:vline,
    markersize=10,
    label="Spikes",
    color=:red,
    alpha=0.6,
)

println("Fitted polynomial coefficients: ", pp_poly.coefficients)
````

## Creating a Custom Intensity Function
For our Gaussian place field, none of the built-in parametric models are ideal given our domain expertise. But do not worry!
With PointProcesses.jl, you can easily create your own custom intensity functions by defining a few methods.
Let's create a custom Gaussian intensity function!

First, we'll define the intensity function structure:

````@example Inhomogeneous
struct GaussianIntensity{R<:Real} <: ParametricIntensity
    peak_rate::R
    center::R
    width::R

    function GaussianIntensity(peak_rate::R, center::R, width::R) where {R<:Real}
        if peak_rate <= 0
            throw(ArgumentError("peak_rate must be positive"))
        end
        if width <= 0
            throw(ArgumentError("width must be positive"))
        end
        return new{R}(peak_rate, center, width)
    end
end
````

Make it callable:

````@example Inhomogeneous
function (f::GaussianIntensity)(t)
    return f.peak_rate * exp(-((t - f.center)^2) / (2 * f.width^2))
end
````

Define how to construct from parameters (for optimization):

````@example Inhomogeneous
function PointProcesses.from_params(
    ::Type{GaussianIntensity{R}}, params::AbstractVector
) where {R}
    peak_rate = exp(params[1])
    center = params[2]
    width = exp(params[3])
    return GaussianIntensity(peak_rate, center, width)
end
````

Define analytical integration (optional but recommended for speed when possible):

````@example Inhomogeneous
function PointProcesses.integrated_intensity(
    f::GaussianIntensity, t_start::T, t_end::T, ::IntegrationConfig
) where {T}
    sqrt_2 = sqrt(2.0)
    prefactor = f.peak_rate * f.width * sqrt(π / 2)

    erf_upper = erf((t_end - f.center) / (sqrt_2 * f.width))
    erf_lower = erf((t_start - f.center) / (sqrt_2 * f.width))

    return prefactor * (erf_upper - erf_lower)
end
````

Now fit our custom Gaussian model:

````@example Inhomogeneous
init_params_gauss = [log(15.0), 5.0, log(2.0)]  #- [log(peak), center, log(width)]

pp_gauss = fit(GaussianIntensity{Float64}, h, init_params_gauss)

println("Fitted Gaussian parameters:") # hide
println("  Peak rate: ", pp_gauss.peak_rate, " Hz") # hide
println("  Center: ", pp_gauss.center) # hide
println("  Width: ", pp_gauss.width) # hide
````

Visualize our custom Gaussian fit:

````@example Inhomogeneous
plot(
    t_range,
    gaussian_place_field.(t_range);
    xlabel="Position",
    ylabel="Firing rate (Hz)",
    label="True intensity",
    linewidth=2,
    title="Custom Gaussian Intensity Fit",
    legend=:topright,
)
plot!(t_range, pp_gauss.(t_range); label="Fitted intensity", linewidth=2, linestyle=:dash)
scatter!(
    h.times,
    zeros(length(h.times));
    marker=:vline,
    markersize=10,
    label="Spikes",
    color=:red,
    alpha=0.6,
)
````

## Model Comparison
Let's compare all models by computing their negative log-likelihoods:

````@example Inhomogeneous
function compute_nll(intensity_func, h)
    nll = -sum(log(intensity_func(t)) for t in h.times)
    config = IntegrationConfig()
    nll += PointProcesses.integrated_intensity(intensity_func, h.tmin, h.tmax, config)
    return nll
end

models = [
    ("Piecewise Constant", pp_piecewise.intensity_function),
    ("Polynomial", pp_poly),
    ("Gaussian (Custom)", pp_gauss),
]

println("\nModel Comparison (Negative Log-Likelihood):") # hide
println("-" ^ 50) # hide
for (name, model) in models # hide
    nll = compute_nll(model, h) # hide
    println("  $name: ", round(nll; digits=2)) # hide
end # hide
````

## Visualizing All Models Together

````@example Inhomogeneous
plot(
    t_range,
    gaussian_place_field.(t_range);
    xlabel="Position",
    ylabel="Firing rate (Hz)",
    label="True intensity",
    linewidth=3,
    title="Comparison of All Fitted Models",
    legend=:topright,
    color=:black,
)

plot!(
    t_range,
    pp_piecewise.intensity_function.(t_range);
    label="Piecewise",
    linewidth=2,
    alpha=0.7,
)
plot!(t_range, pp_poly.(t_range); label="Polynomial", linewidth=2, alpha=0.7)
plot!(
    t_range,
    pp_gauss.(t_range);
    label="Gaussian (Custom)",
    linewidth=2,
    alpha=0.7,
    linestyle=:dash,
)

scatter!(
    h.times,
    zeros(length(h.times));
    marker=:vline,
    markersize=8,
    label="Spikes",
    color=:red,
    alpha=0.5,
)
````

From what we can see the PolynomialIntensity and the custom GaussianIntensity learn an isomorphic representation of the true underlying intensity function.
However, the custom GaussianIntensity has the advantage of interpretability, as its parameters directly correspond to meaningful features of the place field (peak rate, center, width).
This makes it easier to draw conclusions about the neuron's firing behavior based on the fitted model.

