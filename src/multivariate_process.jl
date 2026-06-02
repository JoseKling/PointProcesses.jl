"""
    AbstractMultivariateProcess

Abstract type for multivariate temporal point processes.
"""
abstract type AbstractMultivariateProcess <: AbstractPointProcess end

Base.ndims(pp::AbstractMultivariateProcess) = length(pp.mark_dist)

function mark_distribution(pp::AbstractMultivariateProcess, t, h)
    return [mark_distribution(pp, t, h, d) for d in 1:ndims(pp)]
end

function ground_intensity(pp::AbstractMultivariateProcess, t, h::History)
    return [ground_intensity(pp, t, h, d) for d in 1:ndims(pp)]
end

function integrated_ground_intensity(pp::AbstractMultivariateProcess, h::History, a, b)
    return [integrated_ground_intensity(pp, h, a, b, d) for d in 1:ndims(pp)]
end

function ground_intensity_bound(pp::AbstractMultivariateProcess, t, h::History)
    return [ground_intensity_bound(pp, t, h, d) for d in 1:ndims(pp)]
end

function intensity(pp::AbstractMultivariateProcess, m, t, h, d)
    return ground_intensity(pp, t, h, d) * densityof(pp.mark_dist[d], t, h, m)
end

function intensity(pp::AbstractMultivariateProcess, m, t, h::History)
    return [intensity(pp, m, t, h, d) for d in 1:ndims(pp)]
end

function log_intensity(pp::AbstractMultivariateProcess, m, t, h, d)
    return log(intensity(pp, m, t, h, d))
end

function log_intensity(pp::AbstractMultivariateProcess, m, t, h::History)
    return [log_intensity(pp, m, t, h, d) for d in 1:ndims(pp)]
end

