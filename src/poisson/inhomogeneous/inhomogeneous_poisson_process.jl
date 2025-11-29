"""
    InhomogeneousPoissonProcess{F,M}

Inhomogeneous temporal Poisson process with time-varying intensity.

# Fields

- `intensity_function::F`: callable intensity function λ(t).
- `mark_dist::M`: mark distribution.

# Constructor

    InhomogeneousPoissonProcess(intensity_function, mark_dist)

# Examples

```julia
# Linear intensity
pp = InhomogeneousPoissonProcess(PolynomialIntensity([1.0, 0.5]), Normal())

# Sinusoidal intensity
pp = InhomogeneousPoissonProcess(SinusoidalIntensity(5.0, 2.0, 2π), Categorical([0.3, 0.7]))

# Custom intensity function
pp = InhomogeneousPoissonProcess(t -> 1.0 + 0.5*sin(t), Uniform())
```
"""
struct InhomogeneousPoissonProcess{F,M} <: AbstractPoissonProcess
    intensity_function::F
    mark_dist::M
end

function Base.show(io::IO, pp::InhomogeneousPoissonProcess)
    return print(
        io, "InhomogeneousPoissonProcess($(pp.intensity_function), $(pp.mark_dist))"
    )
end

## AbstractPoissonProcess interface

"""
Mark distribution is time-independent for this implementation.
"""
mark_distribution(pp::InhomogeneousPoissonProcess) = pp.mark_dist

## AbstractPointProcess interface

ground_intensity(pp::InhomogeneousPoissonProcess, t, h) = pp.intensity_function(t)
mark_distribution(pp::InhomogeneousPoissonProcess, t, h) = pp.mark_dist
mark_distribution(pp::InhomogeneousPoissonProcess, t) = pp.mark_dist

function intensity(pp::InhomogeneousPoissonProcess, m, t, h)
    return ground_intensity(pp, t, h) * densityof(mark_distribution(pp, t, h), m)
end

function log_intensity(pp::InhomogeneousPoissonProcess, m, t, h)
    return log(ground_intensity(pp, t, h)) + logdensityof(mark_distribution(pp, t, h), m)
end

"""
    ground_intensity_bound(pp::InhomogeneousPoissonProcess, t, h)

Compute a local upper bound on the ground intensity.

This delegates to `intensity_bound` defined for specific intensity function types
in intensity_methods.jl. Analytical bounds are used when available.
"""
function ground_intensity_bound(pp::InhomogeneousPoissonProcess, t::T, h) where {T}
    return intensity_bound(pp.intensity_function, t)
end

"""
    integrated_ground_intensity(pp::InhomogeneousPoissonProcess, h, a, b)

Compute the integrated ground intensity (compensator) over interval [a, b).

This delegates to `integrated_intensity` defined for specific intensity function types
in intensity_methods.jl. Analytical integrals are used when available.
"""
function integrated_ground_intensity(pp::InhomogeneousPoissonProcess, h, a, b)
    return integrated_intensity(pp.intensity_function, a, b)
end

## Simulation

"""
Override the Poisson-specific rand method to use Ogata's algorithm.

Inhomogeneous processes cannot use the simple Poisson simulation from
`poisson/simulation.jl` since the intensity varies with time.
"""
function Base.rand(
    rng::AbstractRNG, pp::InhomogeneousPoissonProcess, tmin::Real, tmax::Real
)
    return simulate_ogata(rng, pp, tmin, tmax)
end
