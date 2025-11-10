"""
    KSDistance{T<:UnivariateDistribution}

A Kolmogorov-Smirnov distance statistic for testing goodness-of-fit of point processes
against a specified distribution `D` after appropriate time rescaling.

# Type parameter
- `D<:UnivariateDistribution`: the target distribution to test against (e.g., `Exponential`, `Uniform`)

# Available test statistics
- KSDistance(Exponential)
    Kolmogorov-Smirnov distance between the time-changed interevent times and a standard exponential

- KSDistance(Uniform)
    Kolmogorov-Smirnov distance between the time-changed event times and a uniform distribution

# Example
```julia
BootstrapTest(KSDistance{Exponential}, HawkesProcess, hisotry)
```
"""
struct KSDistance{D<:UnivariateDistribution} <: Statistic end

# Convenience method for instantiating a KSDistance type
KSDistance(D::Type{<:UnivariateDistribution}) = KSDistance{D}()

function statistic(::Type{KSDistance{Exponential}}, pp::AbstractPointProcess, h::History)
    (length(h.times) < 2) && return one(typeof(h.tmax)) # If `h` has only 2 elements, than there are no interevent times
    X = diff(time_change(pp, h).times) # X → sorted time re-scaled inter event times
    sort!(X)
    return ksstats(X, Exponential)[2]
end

function statistic(::Type{KSDistance{Uniform}}, pp::AbstractPointProcess, h::History)
    (length(h.times) < 1) && return one(typeof(h.tmax)) # No events ⇒ maximum distance
    transf = time_change(pp, h).times # transf → time re-scaled event times
    return ksstats(transf.times, Uniform(transf.tmin, transf.tmax))[2]
end
