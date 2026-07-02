"""
    AbstractUnivariateProcess

Abstract type for univariate temporal point processes.
"""
abstract type AbstractUnivariateProcess <: AbstractPointProcess end

Base.ndims(::AbstractUnivariateProcess) = 1

function mark_distribution(pp::AbstractUnivariateProcess, t, h::History)
    return mark_distribution(pp.mark_dist, t, h)
end

function sample_mark(rng::AbstractRNG, pp::AbstractPointProcess, t, h::History)
    return sample_mark(rng, pp.mark_dist, t, h)
end
sample_mark(pp::AbstractPointProcess, t, h::History) = sample_mark(default_rng(), pp, t, h)

function intensity(pp::AbstractUnivariateProcess, m, t, h::History)
    return ground_intensity(pp, t, h) * densityof(pp.mark_dist, t, h, m)
end

function log_intensity(pp::AbstractUnivariateProcess, m, t, h::History)
    return log(intensity(pp, m, t, h))
end

"""
    logdensityof(pp, h)

Compute the log probability density function for a temporal point process `pp` applied to history `h`:
```
ℓ(h) = Σₖ log λ(tₖ|hₖ) - Λ(h)
```
The default method uses a loop over events combined with `integrated_ground_intensity`, but it should be reimplemented for specific processes if faster computation is possible.
"""
function DensityInterface.logdensityof(pp::AbstractUnivariateProcess, h::History)
    l = -integrated_ground_intensity(pp, h, min_time(h), max_time(h))
    for (t, m) in zip(event_times(h), event_marks(h))
        l += log_intensity(pp, m, t, h)
    end
    return l
end

function time_change(h::History{T}, pp::AbstractUnivariateProcess) where {T}
    Λ(t) = integrated_ground_intensity(pp, h, min_time(h), t)
    return time_change(h, Λ)
end
