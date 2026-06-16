"""
    IndependentMultivariateProcess{P}

A multivariate point process where each dimension is an independent univariate point process.

# Fields
- `processes::Vector{P}`: vector of univariate point processes, one for each dimension.
"""
struct IndependentMultivariateProcess{P<:AbstractUnivariateProcess} <:
       AbstractMultivariateProcess
    processes::Vector{P}
end

function Base.show(io::IO, pp::IndependentMultivariateProcess)
    return print(io, "IndependentMultivariateProcess($(pp.processes))")
end

Base.ndims(pp::IndependentMultivariateProcess) = length(pp.processes)

## AbstractPointProcess interface
function ground_intensity(pp::IndependentMultivariateProcess, t, h, d)
    return ground_intensity(pp.processes[d], t, History(h, d))
end

function ground_intensity(pp::IndependentMultivariateProcess, t, h)
    return [ground_intensity(pp, t, h, d) for d in 1:ndims(pp)]
end

function mark_distribution(pp::IndependentMultivariateProcess, t, h, d)
    return mark_distribution(pp.processes[d], t, History(h, d))
end

function mark_distribution(pp::IndependentMultivariateProcess, t, h)
    return [mark_distribution(pp, t, h, d) for d in 1:ndims(pp)]
end

function mark_distribution(pp::IndependentMultivariateProcess, t)
    return [mark_distribution(pp.processes[d], t) for d in 1:ndims(pp)]
end

function intensity(pp::IndependentMultivariateProcess, m, t, h, d)
    return intensity(pp.processes[d], m, t, History(h, d))
end

function intensity(pp::IndependentMultivariateProcess, m, t, h)
    return [intensity(pp, m, t, h, d) for d in 1:ndims(pp)]
end

function log_intensity(pp::IndependentMultivariateProcess, m, t, h, d)
    return log(intensity(pp, m, t, h, d))
end

log_intensity(pp::IndependentMultivariateProcess, m, t, h) = log.(intensity(pp, m, t, h))

function ground_intensity_bound(pp::IndependentMultivariateProcess, t, h, d)
    return ground_intensity_bound(pp.processes[d], t, History(h, d))
end

function ground_intensity_bound(pp::IndependentMultivariateProcess, t, h)
    return [ground_intensity_bound(pp, t, h, d) for d in 1:ndims(pp)]
end

function integrated_ground_intensity(pp::IndependentMultivariateProcess, h, a, b, d)
    return integrated_ground_intensity(pp.processes[d], History(h, d), a, b)
end

function integrated_ground_intensity(pp::IndependentMultivariateProcess, h, a, b)
    return [integrated_ground_intensity(pp, h, a, b, d) for d in 1:ndims(pp)]
end

function DensityInterface.logdensityof(pp::IndependentMultivariateProcess, h::History)
    return sum(
        logdensityof(
            pp.processes[d], History(event_times(h, d), h.tmin, h.tmax, event_marks(h, d))
        ) for d in 1:ndims(pp)
    )
end

function time_change(h::History{R,M}, pp::IndependentMultivariateProcess) where {R<:Real,M}
    histories = [time_change(History(h, d), pp.processes[d]) for d in 1:ndims(pp)]
    tmax = maximum(histories[d].tmax for d in 1:ndims(pp))
    return History(
        [histories[d].times for d in 1:ndims(pp)],
        0,
        tmax,
        [histories[d].marks for d in 1:ndims(pp)],
    )
end

function simulate(rng::AbstractRNG, pp::IndependentMultivariateProcess, tmin, tmax)
    histories = [simulate(rng, pp.processes[d], tmin, tmax) for d in 1:ndims(pp)]
    return History(
        [histories[d].times for d in 1:ndims(pp)],
        tmin,
        tmax,
        [histories[d].marks for d in 1:ndims(pp)],
    )
end

function StatsAPI.fit(proc_types::Vector, h::History{R,M}; kwargs...) where {R<:Real,M}
    processes = [
        fit(proc_types[d], History(h, d); kwargs...) for d in eachindex(proc_types)
    ]
    return IndependentMultivariateProcess(processes)
end
