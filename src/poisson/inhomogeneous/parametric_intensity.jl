"""
Trait-based interface for parametric intensity functions.

This provides a clean way to handle parameter transformations for MLE fitting
without passing closures around.
"""

"""
    ParametricIntensity

Abstract trait for intensity functions that can be parameterized for MLE fitting.

Any intensity type that implements this interface must provide:
- `to_params(f)`: Extract parameters from intensity function (for unconstrained optimization)
- `from_params(::Type{F}, params)`: Construct intensity function from parameters
- `initial_params(::Type{F}, h::History)`: Generate initial parameter guess from data

The parameter space should be unconstrained (e.g., use log-transforms for positive parameters).
"""
abstract type ParametricIntensity end

"""
    to_params(f::ParametricIntensity) -> Vector

Extract parameters from an intensity function in unconstrained space.

This should transform constrained parameters (e.g., a > 0) to unconstrained space
(e.g., log(a)) suitable for numerical optimization.
"""
function to_params end

"""
    from_params(::Type{F}, params::AbstractVector) -> F where {F<:ParametricIntensity}

Construct an intensity function from parameters in unconstrained space.

This is the inverse of `to_params`, transforming from unconstrained optimization
space back to the constrained parameter space.
"""
function from_params end

"""
    initial_params(::Type{F}, h::History) -> Vector where {F<:ParametricIntensity}

Generate smart initial parameter guess from event history.

This should provide a reasonable starting point for numerical optimization based
on the observed event times.
"""
function initial_params end
