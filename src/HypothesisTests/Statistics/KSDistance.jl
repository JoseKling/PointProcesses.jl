"""
    KSDistance{T<:UnivariateDistribution}

A Kolmogorov-Smirnov distance statistic for testing goodness-of-fit of point processes
against a specified distribution `D` after appropriate time rescaling.

# Type parameter
- `D<:UnivariateDistribution`: the target distribution to test against (e.g., `Exponential`, `Uniform`)

# Available test statistics
- KSDistance{Exponential}
    Kolmogorov-Smirnov distance between the time-changed interevent times and a standard exponential

- KSDistance{Uniform}
    Kolmogorov-Smirnov distance between the time-changed event times and a uniform distribution

# Example
```julia
BootstrapTest(KSDistance{Exponential}, HawkesProcess, hisotry)
```
"""
struct KSDistance{D<:UnivariateDistribution} <: Statistic end

function statistic(::Type{KSDistance{D}}, pp::AbstractPointProcess, h::History) where {D}
    X, d = transform(D, pp, h) # X → data computed from event history. d → Distribution to compare X to.
    return ksstats(X, d)[2] # `ksstats` always returns a `Float64`
end
