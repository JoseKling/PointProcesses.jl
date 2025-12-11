"""
Trait-based interface for parametric intensity functions.

This provides a clean way to handle parameter transformations for MLE fitting
without passing closures around.
"""

"""
    ParametricIntensity

Abstract trait for intensity functions that can be parameterized for MLE fitting.

Any intensity type that implements this interface must provide:
- `from_vector_params(::Type{F}, params)`: Construct intensity function from parameters

The parameter space should be unconstrained (e.g., use log-transforms for positive parameters).
"""
abstract type ParametricIntensity end

"""
    from_params

Method used in optimization, where paramters are returned as vectors.
Can be used to perform a transformation in the parameter space. Example:
```julia
struct Constant{R} <: ParametricIntensity
    a::R
end where {R<:Real}
(f::Constant, t::Real) = f.a

# Optimization procedure uses this method to calculate the objective.
# This is not constrained to positive paramter values anymore
function from_params(::Constant, params)
    return Constant(exp(params[1]))
end
```
"""
function from_params(F::Type{<:ParametricIntensity}, params; kwargs...)
    return F(params...; kwargs...)
end
