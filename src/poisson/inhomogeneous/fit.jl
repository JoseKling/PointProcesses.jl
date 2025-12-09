"""
Fitting methods for inhomogeneous Poisson processes with various intensity functions.

For parametric intensity functions (PolynomialIntensity, ExponentialIntensity, etc.),
we use maximum likelihood estimation. The log-likelihood is:

    ℓ = ∑ᵢ log(λ(tᵢ)) - ∫ₐᵇ λ(t) dt

For most parametric forms, we use numerical optimization with automatic differentiation.
"""

## Generic MLE infrastructure

"""
    negative_loglikelihood_ipp(params, intensity_type, times, tmin, tmax, integration_config; kwargs...)

Compute the negative log-likelihood for an inhomogeneous Poisson process.

This is the objective function to minimize during MLE. It uses the general form:
    -ℓ(θ) = -∑ᵢ log(λ(tᵢ; θ)) + ∫ₐᵇ λ(t; θ) dt

# Arguments
- `params`: parameter vector to optimize
- `intensity_type`: Type of intensity function (e.g., ExponentialIntensity{Float64})
- `times`: observed event times
- `tmin`, `tmax`: observation window
- `integration_config`: IntegrationConfig for numerical integration
- `kwargs...`: additional keyword arguments for from_params (e.g., link, ω)

# Returns
Negative log-likelihood value (to be minimized).
"""
function negative_loglikelihood_ipp(
    params,
    intensity_type::Type{F},
    times,
    tmin,
    tmax,
    integration_config::IntegrationConfig;
    kwargs...,
) where {F<:ParametricIntensity}
    # Build intensity function from parameters using trait
    # Filter out kwargs that are only for initial_params (like degree)
    from_params_kwargs = filter(kv -> kv[1] ∉ [:degree], collect(pairs(kwargs)))
    intensity_func = from_params(intensity_type, params; from_params_kwargs...)

    # Sum of log intensities at event times (compute first for stability)
    log_sum = zero(eltype(params))
    for t in times
        λ_t = intensity_func(t)
        if λ_t <= 0 || !isfinite(λ_t)
            # Penalize invalid (non-positive or non-finite) intensities heavily
            return typeof(log_sum)(Inf)
        end
        log_sum += log(λ_t)
    end

    # Compute integrated intensity
    Λ = integrated_intensity(intensity_func, tmin, tmax, integration_config)

    # Check for NaN or Inf in integrated intensity
    if !isfinite(Λ)
        return typeof(Λ)(Inf)
    end

    # Return negative log-likelihood
    return Λ - log_sum
end

"""
    fit_mle(
        ::Type{InhomogeneousPoissonProcess{F,M,C}},
        h::History;
        optimizer = LBFGS(),
        autodiff = :forward,
        integration_config = IntegrationConfig(),
        kwargs...
    ) where {F<:ParametricIntensity,M,C}

Fit an inhomogeneous Poisson process using maximum likelihood estimation with numerical optimization.

This version uses the trait-based parameter interface, eliminating the need for manual closures.

# Arguments
- `pptype`: Type of process to fit
- `h`: Event history

# Keyword Arguments
- `optimizer`: Optim.jl optimizer (default: LBFGS())
- `autodiff`: Automatic differentiation mode (default: :forward)
- `integration_config`: IntegrationConfig for numerical integration (default: IntegrationConfig())
- `kwargs...`: additional arguments passed to `from_params` and `initial_params` (e.g., degree, link, ω)

# Returns
Fitted `InhomogeneousPoissonProcess` with optimized parameters.

# Example
```julia
# For ExponentialIntensity: λ(t) = a*exp(b*t)
pp = fit(InhomogeneousPoissonProcess{ExponentialIntensity{Float64}, Normal}, history)

# For PolynomialIntensity with quadratic degree
pp = fit(InhomogeneousPoissonProcess{PolynomialIntensity{Float64}, Normal}, history; degree=2, link=:log)

# With custom integration settings
pp = fit(
    InhomogeneousPoissonProcess{PolynomialIntensity{Float64}, Normal},
    history;
    degree=1,
    integration_config=IntegrationConfig(abstol=1e-10)
)
```
"""
function fit_mle(
    ::Type{InhomogeneousPoissonProcess{F,M,C}},
    h::History;
    optimizer=LBFGS(),
    autodiff=:forward,
    integration_config::C=IntegrationConfig(),
    kwargs...,
) where {F<:ParametricIntensity,M,C}
    times = event_times(h)
    marks = event_marks(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Fit mark distribution independently
    mark_dist = fit(M, marks)

    # Handle empty history: define a literal zero-intensity function
    if isempty(times)
        R = eltype(F) <: Real ? eltype(F) : Float64
        zero_intensity = let z = zero(R)
            t -> z
        end
        return InhomogeneousPoissonProcess(
            zero_intensity, mark_dist; integration_config=integration_config
        )
    end

    # Get initial parameters using trait
    init_params = initial_params(F, h; kwargs...)

    # Define objective function
    objective(params) = negative_loglikelihood_ipp(
        params, F, times, tmin, tmax, integration_config; kwargs...
    )

    result = optimize(objective, init_params, optimizer; autodiff=autodiff)

    # Extract optimal parameters
    optimal_params = Optim.minimizer(result)

    # Build final intensity function using trait
    # Filter out kwargs that are only for initial_params (like degree)
    from_params_kwargs = filter(kv -> kv[1] ∉ [:degree], collect(pairs(kwargs)))
    intensity_func = from_params(F, optimal_params; from_params_kwargs...)

    return InhomogeneousPoissonProcess(
        intensity_func, mark_dist; integration_config=integration_config
    )
end

## PolynomialIntensity fitting (MLE-based)

"""
    fit(::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M,C}}, h::History, degree::Int; link=:log, integration_config=IntegrationConfig()) where {R,M,C}

Fit an inhomogeneous Poisson process with polynomial intensity using MLE.

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history
- `degree`: Degree of the polynomial (0 for constant, 1 for linear, etc.)

# Keyword Arguments
- `link`: Link function, either `:identity` or `:log` (default: `:log` for positivity)
- `integration_config`: IntegrationConfig for numerical integration

Uses maximum likelihood estimation via numerical optimization with automatic differentiation.
The log link is recommended as it ensures positivity of the intensity function.

The mark distribution is fitted separately from the event times.
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M,C}},
    h::History,
    degree::Int;
    link::Symbol=:log,
    integration_config::C=IntegrationConfig(),
) where {R,M,C}
    return fit_mle(
        InhomogeneousPoissonProcess{PolynomialIntensity{R},M,C},
        h;
        degree=degree,
        link=link,
        integration_config=integration_config,
    )
end

"""
    fit(::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M,C}}, h::History; kwargs...) where {R,M,C}

Fit a linear (degree 1) polynomial intensity with log link by default.
"""
function StatsAPI.fit(
    pptype::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M,C}},
    h::History;
    kwargs...,
) where {R,M,C}
    return fit(pptype, h, 1; link=:log, kwargs...)  # Default to linear intensity with log link
end

## ExponentialIntensity fitting (MLE-based)

"""
    fit(::Type{InhomogeneousPoissonProcess{ExponentialIntensity{R},M,C}}, h::History; integration_config=IntegrationConfig()) where {R,M,C}

Fit an inhomogeneous Poisson process with exponential intensity λ(t) = a*exp(b*t) using MLE.

Uses maximum likelihood estimation via numerical optimization with automatic differentiation.
Parameters are transformed to ensure a > 0: internally optimizes log(a).

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history

# Keyword Arguments
- `integration_config`: IntegrationConfig for numerical integration

The mark distribution is fitted separately from the event times.
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{ExponentialIntensity{R},M,C}},
    h::History;
    integration_config::C=IntegrationConfig(),
) where {R,M,C}
    return fit_mle(
        InhomogeneousPoissonProcess{ExponentialIntensity{R},M,C},
        h;
        integration_config=integration_config,
    )
end

## SinusoidalIntensity fitting (MLE-based)

"""
    fit(::Type{InhomogeneousPoissonProcess{SinusoidalIntensity{R},M,C}}, h::History; ω=2π, integration_config=IntegrationConfig()) where {R,M,C}

Fit an inhomogeneous Poisson process with sinusoidal intensity λ(t) = a + b*sin(ω*t + φ) using MLE.

Uses maximum likelihood estimation via numerical optimization with automatic differentiation.
The constraint a >= |b| is enforced by parameterizing: a = exp(p₁), b = tanh(p₂) * exp(p₁).

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history

# Keyword Arguments
- `ω`: Angular frequency (default: 2π for period of 1)
- `integration_config`: IntegrationConfig for numerical integration

The mark distribution is fitted separately from the event times.
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{SinusoidalIntensity{R},M,C}},
    h::History;
    ω::R=R(2π),
    integration_config::C=IntegrationConfig(),
) where {R,M,C}
    return fit_mle(
        InhomogeneousPoissonProcess{SinusoidalIntensity{R},M,C},
        h;
        ω=ω,
        integration_config=integration_config,
    )
end

## PiecewiseConstantIntensity fitting

"""
    fit(::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M,C}}, h::History, n_bins::Int; integration_config=IntegrationConfig()) where {R,M,C}

Fit an inhomogeneous Poisson process with piecewise constant intensity.

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history
- `n_bins`: Number of constant intervals to use

# Keyword Arguments
- `integration_config`: IntegrationConfig for numerical integration

Each bin gets a rate estimated as (count in bin) / (duration of bin).
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M,C}},
    h::History,
    n_bins::Int;
    integration_config::C=IntegrationConfig(),
) where {R,M,C}
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
    return InhomogeneousPoissonProcess(
        intensity_func, mark_dist; integration_config=integration_config
    )
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
