# Type definition
"Abstract type for defining mark distributions not in `Distributions.jl`"
abstract type AbstractMarkDistribution end

const PointProcessMarkDistribution = Union{Distribution,AbstractMarkDistribution}

## Standard implementations. Override as needed
# The docstring for `mark_distribution` lives in `abstract_point_process.jl`, which
# documents both the process and the mark distribution methods of this function.
function mark_distribution(md::AbstractMarkDistribution, t, h::History)
    return error(
        "Type $(typeof(md)) subtypes `AbstractMarkDistribution` but has " *
        "not implemented the required `mark_distribution(md, t, h)` method.",
    )
end

"""
    sample_mark(rng, md, t, h)

Return one sample from the distribution of marks at time `t` after history `h`, using the random number generator `rng`.
"""
function sample_mark(rng::AbstractRNG, md::PointProcessMarkDistribution, t, h::History)
    return rand(rng, mark_distribution(md, t, h))
end

function sample_mark(md::PointProcessMarkDistribution, t, h::History)
    return sample_mark(default_rng(), md, t, h)
end

"The type of the marks returned by the mark distribution"
function Base.eltype(md::AbstractMarkDistribution)
    return typeof(sample_mark(md, 0.0, History(0.0, 1.0, Nothing)))
end

"The likelihood of a mark `m` occurring in an event at time `t` after history `h`"
function DensityInterface.densityof(md::PointProcessMarkDistribution, t, h::History, m)
    return densityof(mark_distribution(md, t, h), m)
end

# Support for `Distributions.jl`
mark_distribution(d::Distribution, t, h::History) = d

StatsAPI.fit(D::Type{<:Distribution}, h::History) = fit(D, h.marks)

# `Type{<:Distribution}` and `Type{<:AbstractPointProcess}` intersect at `Type{Union{}}`,
# because `Union{}` is a subtype of every type. Without this method, inference of
# `fit(PP, h)` for an abstract `PP::Type{<:AbstractPointProcess}` therefore includes the
# return type of the method above (some `Distribution`) in its union, which makes JET
# report a missing method for every call that passes the result on to a process method.
StatsAPI.fit(::Type{Union{}}, ::History) = error("unreachable")

# Struct for non-marked processes
"Mark distribution for non-marked processes. Always return the mark `nothing`."
struct NoMarks <: AbstractMarkDistribution end

mark_distribution(::NoMarks, t, h::History) = Dirac(nothing)

sample_mark(::AbstractRNG, ::NoMarks, t, ::History) = nothing

Base.eltype(::NoMarks) = Nothing

StatsAPI.fit(::Type{NoMarks}, h) = NoMarks()

StatsAPI.fit(::Type{NoMarks}, marks, weights) = NoMarks()

DensityInterface.densityof(::NoMarks, t, ::History, ::Nothing) = 1.0

DensityInterface.densityof(::NoMarks, t, ::History, m) = 0.0
