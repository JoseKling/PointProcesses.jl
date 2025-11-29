"""
Intensity function types for inhomogeneous Poisson processes.

All intensity functions are callable objects that take a time `t` and return λ(t).
"""

"""
    PolynomialIntensity{R<:Real,L}

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
struct PolynomialIntensity{R<:Real,L}
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
    ExponentialIntensity{R<:Real}

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
struct ExponentialIntensity{R<:Real}
    a::R
    b::R

    function ExponentialIntensity(a::R, b::R) where {R<:Real}
        if a <= 0
            throw(ArgumentError("scaling factor 'a' must be positive, got $a"))
        end
        return new{R}(a, b)
    end
end

(f::ExponentialIntensity)(t) = f.a * exp(f.b * t)

function Base.show(io::IO, f::ExponentialIntensity)
    return print(io, "ExponentialIntensity($(f.a), $(f.b))")
end

"""
    SinusoidalIntensity{R<:Real}

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
struct SinusoidalIntensity{R<:Real}
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
