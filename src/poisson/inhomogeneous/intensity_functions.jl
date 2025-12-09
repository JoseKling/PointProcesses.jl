"""
Intensity function types for inhomogeneous Poisson processes.

All intensity functions are callable objects that take a time `t` and return λ(t).
"""

"""
    PolynomialIntensity{R<:Real,L} <: ParametricIntensity

Polynomial intensity function with optional link function.

# Fields

- `coefficients::Vector{R}`: polynomial coefficients [a₀, a₁, ..., aₙ].
- `link::L`: link function applied to the polynomial (`:identity` or `:log`).

# Constructor

    PolynomialIntensity(coefficients; link=:identity)

When `link=:identity`: λ(t) = a₀ + a₁*t + a₂*t² + ... + aₙ*tⁿ
When `link=:log`: λ(t) = exp(a₀ + a₁*t + a₂*t² + ... + aₙ*tⁿ)

The log link ensures positivity of the intensity function.

# Examples

```julia
# Linear identity: λ(t) = 2 + 3*t (may be negative!)
PolynomialIntensity([2.0, 3.0])

# Linear log: λ(t) = exp(2 + 3*t) (always positive)
PolynomialIntensity([2.0, 3.0]; link=:log)

# Quadratic: λ(t) = 1 + 2*t + 0.5*t²
PolynomialIntensity([1.0, 2.0, 0.5])
```
"""
struct PolynomialIntensity{R<:Real,L} <: ParametricIntensity
    coefficients::Vector{R}
    link::L

    function PolynomialIntensity(
        coefficients::Vector{R}; link::Symbol=:identity
    ) where {R<:Real}
        if link ∉ (:identity, :log)
            throw(ArgumentError("link must be :identity or :log, got :$link"))
        end
        return new{R,Symbol}(coefficients, link)
    end
end

function (f::PolynomialIntensity)(t)
    η = f.coefficients[1]
    t_power = one(t)
    for i in 2:length(f.coefficients)
        t_power *= t
        η += f.coefficients[i] * t_power
    end
    return f.link === :log ? exp(η) : η
end

function Base.show(io::IO, f::PolynomialIntensity)
    if f.link === :identity
        return print(io, "PolynomialIntensity($(f.coefficients))")
    else
        return print(io, "PolynomialIntensity($(f.coefficients), link=:$(f.link))")
    end
end

"""
    ExponentialIntensity{R<:Real} <: ParametricIntensity

Exponential intensity function: λ(t) = a*exp(b*t).

# Fields

- `a::R`: scaling factor (must be positive).
- `b::R`: exponential rate.

# Constructor

    ExponentialIntensity(a, b)

# Examples

```julia
# Increasing intensity
ExponentialIntensity(2.0, 0.1)

# Decreasing intensity
ExponentialIntensity(5.0, -0.05)
```
"""
struct ExponentialIntensity{R<:Real} <: ParametricIntensity
    a::R
    b::R

    function ExponentialIntensity(a::R, b::R) where {R<:Real}
        if a <= 0
            throw(ArgumentError("scaling factor 'a' must be positive, got $a"))
        end
        return new{R}(a, b)
    end
end

# Allow mixed types for automatic differentiation compatibility
function ExponentialIntensity(a::A, b::B) where {A<:Real,B<:Real}
    R = promote_type(A, B)
    return ExponentialIntensity(R(a), R(b))
end

(f::ExponentialIntensity)(t) = f.a * exp(f.b * t)

function Base.show(io::IO, f::ExponentialIntensity)
    return print(io, "ExponentialIntensity($(f.a), $(f.b))")
end

"""
    SinusoidalIntensity{R<:Real} <: ParametricIntensity

Sinusoidal intensity function: λ(t) = a + b*sin(ω*t + φ).

To ensure positivity, we require a >= |b| so that λ(t) >= 0 for all t.

# Fields

- `a::R`: baseline intensity (must satisfy a >= |b|).
- `b::R`: amplitude.
- `ω::R`: angular frequency.
- `φ::R`: phase shift.

# Constructor

    SinusoidalIntensity(a, b, ω, φ=0.0)

# Examples

```julia
# Valid: a=5, b=2, so a >= |b|
SinusoidalIntensity(5.0, 2.0, 2π)

# Valid: a=5, b=-3, so a >= |-3| = 3
SinusoidalIntensity(5.0, -3.0, 2π)

# Invalid: a=2, b=3, so a < |b| (will error)
```
"""
struct SinusoidalIntensity{R<:Real} <: ParametricIntensity
    a::R
    b::R
    ω::R
    φ::R

    function SinusoidalIntensity(a::R, b::R, ω::R, φ::R) where {R<:Real}
        if a < abs(b)
            throw(
                ArgumentError(
                    "baseline 'a' must be >= |b| to ensure positivity, got a=$a, b=$b (requires a >= $(abs(b)))",
                ),
            )
        end
        return new{R}(a, b, ω, φ)
    end
end

# Allow mixed types for automatic differentiation compatibility
function SinusoidalIntensity(a::A, b::B, ω::W, φ::P) where {A<:Real,B<:Real,W<:Real,P<:Real}
    R = promote_type(A, B, W, P)
    return SinusoidalIntensity(R(a), R(b), R(ω), R(φ))
end

function SinusoidalIntensity(a::R, b::R, ω::R) where {R<:Real}
    return SinusoidalIntensity(a, b, ω, zero(R))
end

(f::SinusoidalIntensity)(t) = f.a + f.b * sin(f.ω * t + f.φ)

function Base.show(io::IO, f::SinusoidalIntensity)
    return print(io, "SinusoidalIntensity($(f.a), $(f.b), $(f.ω), $(f.φ))")
end

"""
    PiecewiseConstantIntensity{R<:Real}

Piecewise constant intensity function.

# Fields

- `breakpoints::Vector{R}`: sorted vector of breakpoints (including tmin and tmax).
- `rates::Vector{R}`: intensity values for each interval.

# Constructor

    PiecewiseConstantIntensity(breakpoints, rates)

The intensity is `rates[i]` for `t ∈ [breakpoints[i], breakpoints[i+1])`.
"""
struct PiecewiseConstantIntensity{R<:Real}
    breakpoints::Vector{R}
    rates::Vector{R}

    function PiecewiseConstantIntensity(
        breakpoints::Vector{R}, rates::Vector{R}
    ) where {R<:Real}
        if length(rates) != length(breakpoints) - 1
            throw(ArgumentError("rates must have length equal to length(breakpoints) - 1"))
        end
        if !issorted(breakpoints)
            throw(ArgumentError("breakpoints must be sorted"))
        end
        if any(<(0), rates)
            throw(ArgumentError("all rates must be non-negative"))
        end
        return new{R}(breakpoints, rates)
    end
end

function (f::PiecewiseConstantIntensity)(t)
    idx = searchsortedlast(f.breakpoints, t)
    if idx == 0 || idx == length(f.breakpoints)
        return zero(eltype(f.rates))
    end
    return f.rates[idx]
end

function Base.show(io::IO, f::PiecewiseConstantIntensity)
    return print(io, "PiecewiseConstantIntensity($(f.breakpoints), $(f.rates))")
end

"""
    LinearCovariateIntensity{R<:Real,F}

Linear combination of covariate functions: λ(t) = β₀ + β₁*x₁(t) + β₂*x₂(t) + ... + βₙ*xₙ(t).

# Fields

- `intercept::R`: intercept term β₀.
- `coefficients::Vector{R}`: coefficients [β₁, β₂, ..., βₙ].
- `covariates::Vector{F}`: covariate functions [x₁, x₂, ..., xₙ], each callable with signature `xᵢ(t)`.

# Constructor

    LinearCovariateIntensity(intercept, coefficients, covariates)

# Examples

```julia
# With time and sin(time) as covariates
LinearCovariateIntensity(1.0, [0.5, 2.0], [t -> t, t -> sin(t)])

# With custom covariate functions
temp_func = t -> 20 + 5*sin(2π*t/365)  # seasonal temperature
wind_func = t -> 10 + 2*rand()          # wind speed
LinearCovariateIntensity(0.1, [0.05, 0.02], [temp_func, wind_func])
```
"""
struct LinearCovariateIntensity{R<:Real,F}
    intercept::R
    coefficients::Vector{R}
    covariates::Vector{F}

    function LinearCovariateIntensity(
        intercept::R, coefficients::Vector{R}, covariates::Vector{F}
    ) where {R<:Real,F}
        if length(coefficients) != length(covariates)
            throw(ArgumentError("coefficients and covariates must have the same length"))
        end
        return new{R,F}(intercept, coefficients, covariates)
    end
end

function (f::LinearCovariateIntensity)(t)
    η = f.intercept
    for (β, x) in zip(f.coefficients, f.covariates)
        η += β * x(t)
    end
    return exp(η)
end

function Base.show(io::IO, f::LinearCovariateIntensity)
    return print(
        io,
        "LinearCovariateIntensity($(f.intercept), $(f.coefficients), $(length(f.covariates)) covariates)",
    )
end

## ParametricIntensity trait implementations

# PolynomialIntensity
"""
    to_params(f::PolynomialIntensity)

Extract parameters from PolynomialIntensity. Parameters are already unconstrained.
"""
to_params(f::PolynomialIntensity) = f.coefficients

"""
    from_params(::Type{PolynomialIntensity{R}}, params; link=:identity)

Construct PolynomialIntensity from parameters.
"""
function from_params(
    ::Type{PolynomialIntensity{R}}, params::AbstractVector; link::Symbol=:identity
) where {R}
    # Keep params as-is to support ForwardDiff.Dual types
    return PolynomialIntensity(collect(params); link=link)
end

"""
    initial_params(::Type{PolynomialIntensity{R}}, h::History; degree::Int, link=:log)

Generate initial parameter guess for PolynomialIntensity based on event history.
"""
function initial_params(
    ::Type{PolynomialIntensity{R}}, h::History; degree::Int, link::Symbol=:log
) where {R}
    times = event_times(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Compute empirical rate
    empirical_rate = R(length(times)) / (tmax - tmin)

    # Initialize with simple estimates
    if link == :log
        # Start with log of empirical rate for intercept
        # For higher-order terms, use very small values scaled by the time range
        # This prevents overflow when t is large
        intercept = log(empirical_rate)
        higher_order = zeros(R, degree)
        # Scale higher-order coefficients inversely with time range to prevent overflow
        if degree > 0 && tmax > tmin
            scale = 1 / (tmax - tmin)
            # Use very small initial values to avoid exp(large_number) overflow
            higher_order = fill(R(0.001) * scale, degree)
        end
        return vcat([intercept], higher_order)
    else
        # Start with empirical rate for intercept, zeros for higher orders
        return vcat([empirical_rate], zeros(R, degree))
    end
end

# ExponentialIntensity
"""
    to_params(f::ExponentialIntensity)

Extract parameters from ExponentialIntensity in unconstrained space: [log(a), b].
"""
to_params(f::ExponentialIntensity) = [log(f.a), f.b]

"""
    from_params(::Type{ExponentialIntensity{R}}, params)

Construct ExponentialIntensity from unconstrained parameters: params = [log(a), b].
"""
function from_params(::Type{ExponentialIntensity{R}}, params::AbstractVector) where {R}
    return ExponentialIntensity(exp(params[1]), params[2])
end

"""
    initial_params(::Type{ExponentialIntensity{R}}, h::History)

Generate initial parameter guess for ExponentialIntensity based on event history.
"""
function initial_params(::Type{ExponentialIntensity{R}}, h::History) where {R}
    times = event_times(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Assume roughly constant rate to start
    empirical_rate = R(length(times)) / (tmax - tmin)
    return [log(empirical_rate), R(0.0)]  # [log(a), b]
end

# SinusoidalIntensity
"""
    to_params(f::SinusoidalIntensity)

Extract parameters from SinusoidalIntensity in unconstrained space.

Parameters: [log(a), atanh(b/a), φ] where |b/a| < 1 ensures a >= |b|.
"""
function to_params(f::SinusoidalIntensity)
    p1 = log(f.a)
    # atanh may be unstable if b/a ≈ ±1, clamp to safe range
    ratio = clamp(f.b / f.a, -1.0 + eps(eltype(f.a)), 1.0 - eps(eltype(f.a)))
    p2 = atanh(ratio)
    p3 = f.φ
    return [p1, p2, p3]
end

"""
    from_params(::Type{SinusoidalIntensity{R}}, params; ω=2π)

Construct SinusoidalIntensity from unconstrained parameters.

Parameters: [p₁, p₂, p₃] where a = exp(p₁), b = tanh(p₂)*a, φ = p₃.
The ω (angular frequency) must be specified separately.
"""
function from_params(
    ::Type{SinusoidalIntensity{R}}, params::AbstractVector; ω::R=R(2π)
) where {R}
    a = exp(params[1])
    b = tanh(params[2]) * a
    φ = params[3]
    return SinusoidalIntensity(a, b, ω, φ)
end

"""
    initial_params(::Type{SinusoidalIntensity{R}}, h::History; ω=2π)

Generate initial parameter guess for SinusoidalIntensity based on event history.
"""
function initial_params(::Type{SinusoidalIntensity{R}}, h::History; ω::R=R(2π)) where {R}
    times = event_times(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Start with: a = empirical_rate, b = 0, φ = 0
    empirical_rate = R(length(times)) / (tmax - tmin)
    return [log(empirical_rate), R(0.0), R(0.0)]  # [log(a), atanh(b/a), φ]
end
