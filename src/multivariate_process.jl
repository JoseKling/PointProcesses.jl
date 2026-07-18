"""
    AbstractMultivariateProcess

Abstract type for multivariate temporal point processes.

To implement a multivariate process, one must subtype `AbstractMultivariateProcess` and provide
implementations for the methods below. 
- `mark_distribution(pp, t, h, d)`
- `ground_intensity(pp, t, h, d)`
- `integrated_ground_intensity(pp, t, h, d)`
- `ground_intensity_bound(pp, t, h, d)`
- `fit(Type{pp}, h)`
- `simulate(pp, h)`
In all the cases, `pp` is the point process being implemented, `t` is the instant in
which the function will be evaluated, `h` is the history up to `t` and `d` is the dimension.
Other methods should be implemented if permance is an issue.
The process is expected to have a field `mark_dist::Vector{<:AbstractMarkDistribution}`. If this
field is not present, `Base.ndims(pp)` must also be implemented.
"""
abstract type AbstractMultivariateProcess <: AbstractPointProcess end

Base.ndims(pp::AbstractMultivariateProcess) = length(pp.mark_dist)

function mark_distribution(pp::AbstractMultivariateProcess, t, h::History)
    return [mark_distribution(pp, t, h, d) for d in 1:ndims(pp)]
end

function sample_mark(
    rng::AbstractRNG, pp::AbstractMultivariateProcess, t, h::History, d::Int
)
    return sample_mark(rng, pp.mark_dist[d], t, h)
end

function sample_mark(pp::AbstractMultivariateProcess, t, h::History, d::Int)
    return sample_mark(default_rng(), pp, t, h, d)
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

function intensity(pp::AbstractMultivariateProcess, m, t, h::History, d::Int)
    return ground_intensity(pp, t, h, d) * densityof(pp.mark_dist[d], t, h, m)
end

function intensity(pp::AbstractMultivariateProcess, m, t, h::History)
    return [intensity(pp, m, t, h, d) for d in 1:ndims(pp)]
end

function log_intensity(pp::AbstractMultivariateProcess, m, t, h::History, d::Int)
    return log(intensity(pp, m, t, h, d))
end

function log_intensity(pp::AbstractMultivariateProcess, m, t, h::History)
    return [log_intensity(pp, m, t, h, d) for d in 1:ndims(pp)]
end

function time_change(h::History{T}, pp::AbstractMultivariateProcess) where {T}
    tmin = typemax(T)
    tmax = typemin(T)
    transformed_times = [zeros(T, nb_events(h, d)) for d in 1:ndims(h)]
    for d in 1:ndims(pp)
        Λ(t) = integrated_ground_intensity(pp, h, min_time(h), t, d)
        transformed_times[d] .= Λ.(event_times(h, d))
        new_tmin = Λ(min_time(h))
        tmin = new_tmin < tmin ? new_tmin : tmin
        new_tmax = Λ(max_time(h))
        tmax = new_tmax > tmax ? new_tmax : tmax
    end
    transformed_marks = [collect(event_marks(h, d)) for d in 1:ndims(h)]
    return History(transformed_times, tmin, tmax, transformed_marks)
end
