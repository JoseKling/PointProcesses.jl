"""
    PPTest

Interface for all goodness-of-fit tests
"""
abstract type PPTest <: StatsAPI.HypothesisTest end

"""
    pvalue(test::PPTest)

Calculate the p-value of a goodness-of-fit test on a process.

# Arguments
- `bs::BootstrapTest`: the bootstrap test result object

# Returns
- `Float64`: p-value in [0, 1], where small values provide evidence against the null hypothesis
"""
function StatsAPI.pvalue(::PPTest) end

function Base.show(io::IO, t::PPTest)
    print(io, "$(typeof(t)) - pvalue = $(pvalue(t))")
end

#=
Internal function for performing the appropriate transformation
on the event times according to the selected distribution.
=#
function transform(::Type{<:Uniform}, pp::AbstractPointProcess, h)
    (length(h.times) < 1) && return 1.0 # No events ⇒ maximum distance
    transf = time_change(h, pp) # transf → time re-scaled event times
    return transf.times, Uniform(transf.tmin, transf.tmax)
end

function transform(::Type{<:Exponential}, pp::AbstractPointProcess, h)
    (length(h.times) < 2) && return 1.0 # If `h` has only 2 elements, than there are no interevent times
    inter_transf = diff(time_change(h, pp).times) # inter_transf → sorted time transformed inter event times
    sort!(inter_transf)
    return inter_transf, Exponential(1)
end
