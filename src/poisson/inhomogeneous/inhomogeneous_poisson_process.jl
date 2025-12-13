"""
    InhomogeneousPoissonProcess{F,M,C}

Inhomogeneous temporal Poisson process with time-varying intensity.

# Fields

- `intensity_function::F`: callable intensity function λ(t).
- `mark_dist::M`: mark distribution.
- `integration_config::C`: configuration for numerical integration.

# Constructor

    InhomogeneousPoissonProcess(intensity_function, mark_dist; integration_config=IntegrationConfig())

# Examples

```julia
# Linear intensity
pp = InhomogeneousPoissonProcess(PolynomialIntensity([1.0, 0.5]), Normal())

# Sinusoidal intensity
pp = InhomogeneousPoissonProcess(SinusoidalIntensity(5.0, 2.0, 2π), Categorical([0.3, 0.7]))

# Custom intensity function
pp = InhomogeneousPoissonProcess(t -> 1.0 + 0.5*sin(t), Uniform())

# Custom integration settings
pp = InhomogeneousPoissonProcess(
    my_intensity,
    Normal(),
    integration_config=IntegrationConfig(abstol=1e-10)
)
```
"""
struct InhomogeneousPoissonProcess{F,M,C} <: AbstractPointProcess
    intensity_function::F
    mark_dist::M
    integration_config::C
end

# Main constructor with default config
function InhomogeneousPoissonProcess(
    f::F, mark_dist::M; integration_config::C=IntegrationConfig()
) where {F,M,C}
    return InhomogeneousPoissonProcess{F,M,C}(f, mark_dist, integration_config)
end

# Convenience constructor for no marks
function InhomogeneousPoissonProcess(
    f::F; integration_config::C=IntegrationConfig()
) where {F,C}
    return InhomogeneousPoissonProcess{F,Dirac{Nothing},C}(
        f, Dirac(nothing), integration_config
    )
end

function Base.show(io::IO, pp::InhomogeneousPoissonProcess)
    return print(
        io, "InhomogeneousPoissonProcess($(pp.intensity_function), $(pp.mark_dist))"
    )
end

## Access methods

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

"""
    ground_intensity_bound(pp::InhomogeneousPoissonProcess, t, h)

Compute a local upper bound on the ground intensity.

This delegates to `intensity_bound` defined for specific intensity function types
in intensity_methods.jl. Analytical bounds are used when available.
"""
function ground_intensity_bound(pp::InhomogeneousPoissonProcess, t::T, h::History) where {T}
    return intensity_bound(pp.intensity_function, t, h)
end

"""
    integrated_ground_intensity(pp::InhomogeneousPoissonProcess, h, a, b)

Compute the integrated ground intensity (compensator) over interval [a, b).

This delegates to `integrated_intensity` defined for specific intensity function types
in intensity_methods.jl. Analytical integrals are used when available, otherwise
numerical integration is performed using the configuration stored in `pp.integration_config`.
"""
function integrated_ground_intensity(
    pp::InhomogeneousPoissonProcess, h::History, t_start::T, t_end::T
) where {T}
    return integrated_intensity(
        pp.intensity_function, t_start, t_end, pp.integration_config
    )
end

## Simulation
function simulate(rng::AbstractRNG, pp::InhomogeneousPoissonProcess, tmin::Real, tmax::Real)
    return simulate_ogata(rng, pp, tmin, tmax)
end
