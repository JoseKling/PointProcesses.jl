"""
    IntegrationConfig

Configuration for numerical integration of intensity functions.

# Fields

- `solver`: Integration solver from Integrals.jl (e.g., QuadGKJL(), HCubatureJL())
- `abstol::Float64`: Absolute tolerance for integration
- `reltol::Float64`: Relative tolerance for integration
- `maxiters::Int`: Maximum number of iterations

# Constructor

    IntegrationConfig(; solver=QuadGKJL(), abstol=1e-8, reltol=1e-8, maxiters=1000)

# Examples

```julia
# Default configuration
config = IntegrationConfig()

# Higher precision
config = IntegrationConfig(abstol=1e-12, reltol=1e-12)

# Different solver
using Integrals
config = IntegrationConfig(solver=HCubatureJL())
```
"""
struct IntegrationConfig
    solver::Any
    abstol::Float64
    reltol::Float64
    maxiters::Int

    function IntegrationConfig(;
        solver=QuadGKJL(), abstol::Float64=1e-8, reltol::Float64=1e-8, maxiters::Int=1000
    )
        new(solver, abstol, reltol, maxiters)
    end
end

function Base.show(io::IO, config::IntegrationConfig)
    return print(
        io,
        "IntegrationConfig(solver=$(typeof(config.solver)), abstol=$(config.abstol), reltol=$(config.reltol), maxiters=$(config.maxiters))",
    )
end
