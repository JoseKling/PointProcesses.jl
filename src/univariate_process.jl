"""
    AbstractUnivariateProcess

Abstract type for univariate temporal point processes.
"""
abstract type AbstractUnivariateProcess <: AbstractPointProcess end

Base.ndims(::AbstractUnivariateProcess) = 1

function mark_distribution(pp::AbstractUnivariateProcess, t, h)
    mark_distribution(pp.mark_dist, t, h)
end

function intensity(pp::AbstractUnivariateProcess, m, t, h)
    return ground_intensity(pp, t, h) * densityof(pp.mark_dist, t, h, m)
end

function log_intensity(pp::AbstractUnivariateProcess, m, t, h)
    return log(intensity(pp, m, t, h))
end

function DensityInterface.logdensityof(pp::AbstractUnivariateProcess, h::History)
    l = -integrated_ground_intensity(pp, h, min_time(h), max_time(h))
    for (t, m) in zip(event_times(h), event_marks(h))
        l += log_intensity(pp, m, t, h)
    end
    return l
end

function time_change(h::History, pp::AbstractUnivariateProcess)
    Λ(t) = integrated_ground_intensity(pp, h, min_time(h), t)
    return time_change(h, Λ)
end
