"""
Intensity function types for inhomogeneous Poisson processes.

All intensity functions are callable objects that take a time `t` and return λ(t).
"""

"""
    PolynomialIntensity{R<:Real}

Polynomial intensity function: λ(t) = a₀ + a₁*t + a₂*t² + ... + aₙ*tⁿ.

# Fields

- `coefficients::Vector{R}`: polynomial coefficients [a₀, a₁, ..., aₙ].

# Constructor

    PolynomialIntensity(coefficients)

# Examples

```julia
# Linear: λ(t) = 2 + 3*t
PolynomialIntensity([2.0, 3.0])

# Quadratic: λ(t) = 1 + 2*t + 0.5*t²
PolynomialIntensity([1.0, 2.0, 0.5])
```
"""
struct PolynomialIntensity{R<:Real}
    coefficients::Vector{R}
end

function (f::PolynomialIntensity)(t)
    result = f.coefficients[1]
    t_power = one(t)
    for i in 2:length(f.coefficients)
        t_power *= t
        result += f.coefficients[i] * t_power
    end
    return result
end

function Base.show(io::IO, f::PolynomialIntensity)
    return print(io, "PolynomialIntensity($(f.coefficients))")
end

"""
    ExponentialIntensity{R<:Real}

Exponential intensity function: λ(t) = a*exp(b*t).

# Fields

- `a::R`: scaling factor.
- `b::R`: exponential rate.

# Constructor

    ExponentialIntensity(a, b)
"""
struct ExponentialIntensity{R<:Real}
    a::R
    b::R
end

(f::ExponentialIntensity)(t) = f.a * exp(f.b * t)

function Base.show(io::IO, f::ExponentialIntensity)
    return print(io, "ExponentialIntensity($(f.a), $(f.b))")
end

"""
    SinusoidalIntensity{R<:Real}

Sinusoidal intensity function: λ(t) = a + b*sin(ω*t + φ).

# Fields

- `a::R`: baseline intensity.
- `b::R`: amplitude.
- `ω::R`: angular frequency.
- `φ::R`: phase shift.

# Constructor

    SinusoidalIntensity(a, b, ω, φ=0.0)
"""
struct SinusoidalIntensity{R<:Real}
    a::R
    b::R
    ω::R
    φ::R
end

function SinusoidalIntensity(a::R, b::R, ω::R) where {R<:Real}
    SinusoidalIntensity(a, b, ω, zero(R))
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
    result = f.intercept
    for (β, x) in zip(f.coefficients, f.covariates)
        result += β * x(t)
    end
    return result
end

function Base.show(io::IO, f::LinearCovariateIntensity)
    return print(
        io,
        "LinearCovariateIntensity($(f.intercept), $(f.coefficients), $(length(f.covariates)) covariates)",
    )
end
