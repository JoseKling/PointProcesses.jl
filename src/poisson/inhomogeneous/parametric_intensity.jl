"""
Trait-based interface for parametric intensity functions.

This provides a clean way to handle parameter transformations for MLE fitting
without passing closures around.
"""

"""
    ParametricIntensity

Abstract trait for intensity functions that can be parameterized for MLE fitting.

Any intensity type that implements this interface must provide:
- `from_params(::Type{F}, params)`: Construct intensity function from parameters
- `initial_params(::Type{F}, h::History)`: Generate initial parameter guess from data

The parameter space should be unconstrained (e.g., use log-transforms for positive parameters).
"""
abstract type ParametricIntensity end

"""
    from_params(::Type{F}, params::AbstractVector) -> F where {F<:ParametricIntensity}

Construct an intensity function from parameters in unconstrained space.

Parameters are transformed from unconstrained optimization space to the constrained
parameter space (e.g., exp(p) for positive parameters).
"""
function from_params end

"""
    initial_params(::Type{F}, h::History) -> Vector where {F<:ParametricIntensity}

Generate smart initial parameter guess from event history.

This should provide a reasonable starting point for numerical optimization based
on the observed event times.
"""
function initial_params end
