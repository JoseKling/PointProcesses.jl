"""
    PointProcessTest

Interface for all goodness-of-fit tests
"""
abstract type PointProcessTest <: StatsAPI.HypothesisTest end

"""
    pvalue(test::PointProcessTest)

Calculate the p-value of a goodness-of-fit test on a process.

# Arguments
- `::PointProcessTest`: the test result object

# Returns
- `Float64`: p-value in [0, 1], where small values provide evidence against the null hypothesis
"""
function StatsAPI.pvalue(::PointProcessTest) end

function Base.show(io::IO, t::PointProcessTest)
    print(io, "$(typeof(t)) - pvalue = $(pvalue(t))")
end

#=
Internal function for performing the appropriate transformation
on the event times according to the selected distribution.
=#
function transform(::Type{<:Uniform}, pp::AbstractPointProcess, h)
    transf = time_change(h, pp) # transf → time re-scaled event times
    return transf.times, Uniform(transf.tmin, transf.tmax)
end

function transform(::Type{<:Exponential}, pp::AbstractPointProcess, h)
    inter_transf = diff(time_change(h, pp).times) # inter_transf → sorted time transformed inter event times
    sort!(inter_transf)
    return inter_transf, Exponential(1)
end
