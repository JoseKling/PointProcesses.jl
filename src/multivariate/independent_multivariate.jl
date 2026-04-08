struct IndependentMultivariateProcess
    processes::Vector{<:AbstractUnivariateProcess}
end

function Base.show(io::IO, pp::IndependentMultivariateProcess)
    return print(io, "IndependentMultivariateProcess($(pp.processes))")
end

Base.ndims(pp::IndependentMultivariateProcess) = length(pp.processes)

## AbstractPointProcess interface
function ground_intensity(pp::IndependentMultivariateProcess, t, h, d)
    return ground_intensity(pp.processes[d], t, h)
end

function ground_intensity(pp::IndependentMultivariateProcess, t, h)
    return [ground_intensity(pp.processes, t, h, d) for d in eachindex(pp.processes)]
end

function mark_distribution(pp::IndependentMultivariateProcess, t, h, d)
    return mark_distribution(pp.processes[d], t, h)
end

function mark_distribution(pp::IndependentMultivariateProcess, t, h)
    return [mark_distribution(pp.processes, t, h, d) for d in eachindex(pp.processes)]
end

function intensity(pp::IndependentMultivariateProcess, m, t, h, d) 
    return intensity(pp.processes[d], m, t, h)
end

function intensity(pp::IndependentMultivariateProcess, m, t, h) 
    return [intensity(pp, m, t, h, d) for d in eachindex(pp.processes)]
end

log_intensity(pp::IndependentMultivariateProcess, m, t, h, d) = log(intensity(pp.processes, m, t, h, d))
log_intensity(pp::IndependentMultivariateProcess, m, t, h) = log.(intensity(pp, m, t, h))

function ground_intensity_bound(pp::IndependentMultivariateProcess, t, h, d)
    return ground_intensity_bound(pp.processes[d], t, h)
end

function ground_intensity_bound(pp::IndependentMultivariateProcess, t, h)
    return [ground_intensity_bound(pp, t, h, d) for d in eachindex(pp.processes)]
end

function integrated_ground_intensity(pp::IndependentMultivariateProcess, h, a, b, d)
    return integrated_ground_intensity(pp.processes[d], h, a, b)
end

function integrated_ground_intensity(pp::IndependentMultivariateProcess, h, a, b)
    return [integrated_ground_intensity(pp, h, a, b, d) for d in eachindex(pp.processes)]
end

function time_change(h::History{R,M}, pp::IndependentMultivariateProcess) where {R<:Real,M}
    histories = [time_change(h, pp.processes[d]) for d in eachindex(pp.processes)]
    tmax = maximum(histories[d].tmax for d in eachindex(pp.processes))
    return History(
        [histories[d].times for d in eachindex(pp.processes)],
        0,
        tmax,
        [histories[d].marks for d in eachindex(pp.processes)]
    )
end

function simulate(pp::IndependentMultivariateProcess, tmin, tmax)
    histories = [simulate(pp.processes[d], tmin, tmax) for d in eachindex(pp.processes)]
    return History(
        [histories[d].times for d in eachindex(pp.processes)],
        tmin,
        tmax,
        [histories[d].marks for d in eachindex(pp.processes)]
    )
end

function StatsAPI.fit(v::Vector{DataType}, h::History{R,M}; kwargs...) where {R<:Real,M}
    if any(!(x <: AbstractUnivariateProcess) for x in v)
        throw(ArgumentError("All elements of v must be subtypes of AbstractUnivariateProcess."))
    end
    processes = [fit(v[d], History(event_times(h, d), h.tmin, h.tmax, event_marks(h, d)); kwargs...) for d in eachindex(v)]
    return IndependentMultivariateProcess(processes)
end