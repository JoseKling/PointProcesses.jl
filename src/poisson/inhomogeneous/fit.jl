"""
Fitting methods for inhomogeneous Poisson processes with various intensity functions.

For parametric intensity functions (PolynomialIntensity, ExponentialIntensity, etc.),
we use maximum likelihood estimation. The log-likelihood is:

    ℓ = ∑ᵢ log(λ(tᵢ)) - ∫ₐᵇ λ(t) dt

For most parametric forms, we use numerical optimization with automatic differentiation.
"""

## Generic MLE infrastructure

"""
    negative_loglikelihood_ipp(params, param_to_intensity, times, tmin, tmax, mark_dist)

Compute the negative log-likelihood for an inhomogeneous Poisson process.

This is the objective function to minimize during MLE. It uses the general form:
    -ℓ(θ) = -∑ᵢ log(λ(tᵢ; θ)) + ∫ₐᵇ λ(t; θ) dt

# Arguments
- `params`: parameter vector to optimize
- `param_to_intensity`: function that converts params to an intensity function
- `times`: observed event times
- `tmin`, `tmax`: observation window
- `mark_dist`: mark distribution (used for output only, fitted separately)

# Returns
Negative log-likelihood value (to be minimized).
"""
function negative_loglikelihood_ipp(params, param_to_intensity, times, tmin, tmax)
    # Build intensity function from parameters
    intensity_func = param_to_intensity(params)

    # Compute integrated intensity (compensator)
    Λ = integrated_intensity(intensity_func, tmin, tmax)

    # Sum of log intensities at event times
    log_sum = zero(eltype(params))
    for t in times
        λ_t = intensity_func(t)
        if λ_t <= 0
            # Penalize invalid (non-positive) intensities heavily
            return typeof(log_sum)(Inf)
        end
        log_sum += log(λ_t)
    end

    # Return negative log-likelihood
    return Λ - log_sum
end

"""
    fit_mle(
        ::Type{InhomogeneousPoissonProcess{F,M}},
        h::History,
        param_to_intensity::Function,
        intensity_to_params::Function,
        initial_params::Vector;
        optimizer = LBFGS(),
        autodiff = :forward
    ) where {F,M}

Fit an inhomogeneous Poisson process using maximum likelihood estimation with numerical optimization.

# Arguments
- `pptype`: Type of process to fit
- `h`: Event history
- `param_to_intensity`: Function mapping parameter vector to intensity function
- `intensity_to_params`: Function extracting parameters from an intensity function (for initialization)
- `initial_params`: Initial parameter values for optimization

# Keyword Arguments
- `optimizer`: Optim.jl optimizer (default: LBFGS())
- `autodiff`: Automatic differentiation mode (default: :forward)

# Returns
Fitted `InhomogeneousPoissonProcess` with optimized parameters.

# Example
```julia
# For ExponentialIntensity: λ(t) = a*exp(b*t)
param_to_intensity = p -> ExponentialIntensity(exp(p[1]), p[2])
intensity_to_params = f -> [log(f.a), f.b]
initial_params = [log(2.0), 0.1]

pp = fit_mle(
    InhomogeneousPoissonProcess{ExponentialIntensity{Float64}, Normal},
    history,
    param_to_intensity,
    intensity_to_params,
    initial_params
)
```
"""
function fit_mle(
    ::Type{InhomogeneousPoissonProcess{F,M}},
    h::History,
    param_to_intensity::Function,
    intensity_to_params::Function,
    initial_params::Vector;
    optimizer=LBFGS(),
    autodiff=:forward,
) where {F,M}
    times = event_times(h)
    marks = event_marks(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Fit mark distribution independently
    mark_dist = fit(M, marks)

    # Handle empty history
    if isempty(times)
        # Return a process with zero intensity
        intensity_func = param_to_intensity(zero(initial_params))
        return InhomogeneousPoissonProcess(intensity_func, mark_dist)
    end

    # Define objective function
    objective(params) = negative_loglikelihood_ipp(
        params, param_to_intensity, times, tmin, tmax
    )

    # Optimize using Optim.jl with automatic differentiation
    result = optimize(objective, initial_params, optimizer; autodiff=autodiff)

    # Extract optimal parameters
    optimal_params = Optim.minimizer(result)

    # Build final intensity function
    intensity_func = param_to_intensity(optimal_params)

    return InhomogeneousPoissonProcess(intensity_func, mark_dist)
end

## PolynomialIntensity fitting (MLE-based)

"""
    fit(::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}}, h::History, degree::Int; link=:log) where {R,M}

Fit an inhomogeneous Poisson process with polynomial intensity using MLE.

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history
- `degree`: Degree of the polynomial (0 for constant, 1 for linear, etc.)
- `link`: Link function, either `:identity` or `:log` (default: `:log` for positivity)

Uses maximum likelihood estimation via numerical optimization with automatic differentiation.
The log link is recommended as it ensures positivity of the intensity function.

The mark distribution is fitted separately from the event times.
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}},
    h::History,
    degree::Int;
    link::Symbol=:log,
) where {R,M}
    times = event_times(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Initialize with simple estimates
    # For log link: start with small values
    # For identity link: start with empirical rate
    if link == :log
        initial_params = vcat([log(R(length(times)) / (tmax - tmin))], zeros(R, degree))
    else
        initial_params = vcat([R(length(times)) / (tmax - tmin)], zeros(R, degree))
    end

    # Define parameter transformation functions
    param_to_intensity(p) = PolynomialIntensity(p; link=link)
    intensity_to_params(f) = f.coefficients

    return fit_mle(
        InhomogeneousPoissonProcess{PolynomialIntensity{R},M},
        h,
        param_to_intensity,
        intensity_to_params,
        initial_params,
    )
end

"""
    fit(::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}}, h::History) where {R,M}

Fit a linear (degree 1) polynomial intensity with log link by default.
"""
function StatsAPI.fit(
    pptype::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}}, h::History
) where {R,M}
    return fit(pptype, h, 1; link=:log)  # Default to linear intensity with log link
end

## ExponentialIntensity fitting (MLE-based)

"""
    fit(::Type{InhomogeneousPoissonProcess{ExponentialIntensity{R},M}}, h::History) where {R,M}

Fit an inhomogeneous Poisson process with exponential intensity λ(t) = a*exp(b*t) using MLE.

Uses maximum likelihood estimation via numerical optimization with automatic differentiation.
Parameters are transformed to ensure a > 0: internally optimizes log(a).

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history

The mark distribution is fitted separately from the event times.
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{ExponentialIntensity{R},M}}, h::History
) where {R,M}
    times = event_times(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Initialize with reasonable estimates
    # Assume roughly constant rate to start
    empirical_rate = R(length(times)) / (tmax - tmin)
    initial_params = [log(empirical_rate), R(0.0)]  # [log(a), b]

    # Parameter transformation: params = [log(a), b]
    # This ensures a > 0
    param_to_intensity(p) = ExponentialIntensity(exp(p[1]), p[2])
    intensity_to_params(f) = [log(f.a), f.b]

    return fit_mle(
        InhomogeneousPoissonProcess{ExponentialIntensity{R},M},
        h,
        param_to_intensity,
        intensity_to_params,
        initial_params,
    )
end

## SinusoidalIntensity fitting (MLE-based)

"""
    fit(::Type{InhomogeneousPoissonProcess{SinusoidalIntensity{R},M}}, h::History; ω=2π) where {R,M}

Fit an inhomogeneous Poisson process with sinusoidal intensity λ(t) = a + b*sin(ω*t + φ) using MLE.

Uses maximum likelihood estimation via numerical optimization with automatic differentiation.
The constraint a >= |b| is enforced by parameterizing: a = exp(p₁), b = tanh(p₂) * exp(p₁).

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history
- `ω`: Angular frequency (default: 2π for period of 1)

The mark distribution is fitted separately from the event times.
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{SinusoidalIntensity{R},M}}, h::History; ω::R=R(2π)
) where {R,M}
    times = event_times(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Initialize with reasonable estimates
    empirical_rate = R(length(times)) / (tmax - tmin)
    # Start with: a = empirical_rate, b = 0, φ = 0
    initial_params = [log(empirical_rate), R(0.0), R(0.0)]  # [log(a), atanh(b/a), φ]

    # Parameter transformation to enforce a >= |b|:
    # params = [p₁, p₂, p₃] where a = exp(p₁), b = tanh(p₂) * exp(p₁), φ = p₃
    # This ensures |b| <= a since |tanh(p₂)| <= 1
    function param_to_intensity(p)
        a = exp(p[1])
        b = tanh(p[2]) * a
        φ = p[3]
        return SinusoidalIntensity(a, b, ω, φ)
    end

    function intensity_to_params(f)
        p1 = log(f.a)
        p2 = atanh(f.b / f.a)  # Note: may be unstable if b/a ≈ ±1
        p3 = f.φ
        return [p1, p2, p3]
    end

    return fit_mle(
        InhomogeneousPoissonProcess{SinusoidalIntensity{R},M},
        h,
        param_to_intensity,
        intensity_to_params,
        initial_params,
    )
end

## PiecewiseConstantIntensity fitting

"""
    fit(::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}}, h::History, n_bins::Int) where {R,M}

Fit an inhomogeneous Poisson process with piecewise constant intensity.

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history
- `n_bins`: Number of constant intervals to use

Each bin gets a rate estimated as (count in bin) / (duration of bin).
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}},
    h::History,
    n_bins::Int,
) where {R,M}
    times = event_times(h)
    marks = event_marks(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Fit mark distribution
    mark_dist = fit(M, marks)

    # Create bins
    breakpoints = collect(range(R(tmin), R(tmax); length=n_bins + 1))

    # Count events in each bin
    counts = zeros(Int, n_bins)
    for t in times
        bin_idx = searchsortedlast(breakpoints, t)
        bin_idx = max(1, min(bin_idx, n_bins))
        counts[bin_idx] += 1
    end

    # Compute rates for each bin
    bin_duration = (tmax - tmin) / n_bins
    rates = R.(counts ./ bin_duration)

    intensity_func = PiecewiseConstantIntensity(breakpoints, rates)
    return InhomogeneousPoissonProcess(intensity_func, mark_dist)
end

"""
    fit(::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}}, h::History) where {R,M}

Fit with 10 bins by default.
"""
function StatsAPI.fit(
    pptype::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}}, h::History
) where {R,M}
    return fit(pptype, h, 10)
end

## Helper methods for multiple histories

function StatsAPI.fit(
    pptype::Type{<:InhomogeneousPoissonProcess},
    histories::AbstractVector{<:History},
    args...,
)
    # Concatenate all histories
    combined = histories[1]
    for i in 2:length(histories)
        combined = History(
            vcat(event_times(combined), event_times(histories[i])),
            vcat(event_marks(combined), event_marks(histories[i])),
            min(min_time(combined), min_time(histories[i])),
            max(max_time(combined), max_time(histories[i])),
        )
    end
    return fit(pptype, combined, args...)
end
