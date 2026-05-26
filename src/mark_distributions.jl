
# Type definition
"Abstract type for defining mark distributions not in `Distributions.jl`"
abstract type AbstractMarkDistribution end

const PointProcessMarkDistribution = Union{Distribution,AbstractMarkDistribution}

## Standard implementations. Override as needed
"""
    mark_distribution(md, t, h)

Compute the distribution of marks at time `t` after history `h`.
"""
function mark_distribution end

function mark_distribution(md::AbstractMarkDistribution, t, h::History)
    error(
        "Type $(typeof(md)) subtypes `AbstractMarkDistribution` but has " *
        "not implemented the required `mark_distribution(md, t, h)` method.",
    )
end

"""
    sample_mark(rng, md, t, h)

Return one sample from the distribution of marks at time `t` after history `h`, using the random number generator `rng`.
"""
function sample_mark(rng::AbstractRNG, md::PointProcessMarkDistribution, t, h::History)
    rand(rng, mark_distribution(md, t, h))
end

function sample_mark(md::PointProcessMarkDistribution, t, h::History)
    sample_mark(default_rng(), md, t, h)
end

"The type of the marks returned by the mark distribution"
function Base.eltype(md::AbstractMarkDistribution)
    eltype(mark_distribution(md, 0.0, History(0.0, 1.0)))
end

"The likelihood of a mark `m` occurring in an event at time `t` after history `h`"
function DensityInterface.densityof(md::PointProcessMarkDistribution, t, h::History, m)
    densityof(mark_distribution(md, t, h), m)
end

StatsAPI.fit

# Support for `Distributions.jl`
mark_distribution(d::Distribution, t, h::History) = d

StatsAPI.fit(D::Type{<:Distribution}, h::History) = fit(D, h.marks)

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
