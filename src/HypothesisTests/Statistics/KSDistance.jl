"""
    KSDistance{T<:UnivariateDistribution}

A Kolmogorov-Smirnov distance statistic for testing goodness-of-fit of point processes
against a specified distribution `D` after appropriate time rescaling.

# Type parameter

- `D<:UnivariateDistribution`: the target distribution to test against (e.g., `Exponential`, `Uniform`)

# Example
```julia
BootstrapTest(KSDistance{Exponential}, HawkesProcess, hisotry)
```
"""
struct KSDistance{D<:UnivariateDistribution} end
KSDistance(D::Type{<:UnivariateDistribution}) = KSDistance{D}()

"""
    statistic(::KSDistance{Exponential}, pp::AbstractPointProcess, h::History) -> Float64

Compute the Kolmogorov-Smirnov distance between time-rescaled inter-event times 
and the standard exponential distribution.

# Arguments

- `::KSDistance{Exponential}`: the KS statistic for exponential distribution
- `pp::AbstractPointProcess`: the point process model
- `h::History`: the observed event history

# Returns

- `Float64`: the KS distance statistic (maximum absolute difference between empirical and theoretical CDFs)

# Notes

- Returns 1.0 if there are fewer than 2 events (no inter-event times available)

# Example

```julia
ks_stat = statistic(KSDistance(Exponential), hawkes_process, history)
```
"""
function statistic(::KSDistance{Exponential}, pp::AbstractPointProcess, h::History)
    (length(h.times) < 2) && return 1.0 # If `h` has only 2 elements, than there are no interevent times
    X = diff(time_change(pp, h).times) # X → sorted time re-scaled inter event times
    sort!(X)
    return ksstats(X, Exponential)[2]
end

"""
    statistic(::KSDistance{Uniform}, pp::AbstractPointProcess, h::History) -> Float64

Compute the Kolmogorov-Smirnov distance between time-rescaled event times 
and the uniform distribution on the transformed time interval [0, Λ(T)].

# Arguments

- `::KSDistance{Uniform}`: the KS statistic for uniform distribution
- `pp::AbstractPointProcess`: the point process model  
- `h::History`: the observed event history

# Returns

- `Float64`: the KS distance statistic (maximum absolute difference between empirical and theoretical CDFs)

# Notes

- Returns 1.0 if there are no events

# Example

```julia
s = statistic(KSDistance(Uniform), HawkesProcess, history)
```
"""
function statistic(::KSDistance{Uniform}, pp::AbstractPointProcess, h::History)
    (length(h.times) < 1) && return 1.0 # No events ⇒ maximum distance
    transf = time_change(pp, h).times # transf → time re-scaled event times
    return ksstats(transf.times, Uniform(transf.tmin, transf.tmax))[2]
end

