"""
    PoissonProcess{R,D}

Homogeneous temporal Poisson process with arbitrary mark distribution.

# Fields

- `λ::R`: ground intensity.
- `mark_dist::D`: mark distribution.

# Constructor

    PoissonProcess(λ, mark_dist)
"""
struct PoissonProcess{R<:Real,D} <: AbstractPointProcess
    λ::Vector{R}
    mark_dist::Vector{D}

    function PoissonProcess(λ::Vector{R}, mark_dist::Vector{D}; check_args::Bool=true) where {R<:Real,D}
        if check_args
            all(λ .>= zero(λ)) ||
            throw(
                DomainError(
                    λ, "PoissonProcess: the ground intensity λ must be non negative."
                ),
            )
            length(λ) == length(mark_dist) ||
            throw(
                DimensionMismatch(
                    "λ = $λ, mark_dist = $mark_dist.\nPoissonProcess: the length of λ and mark_dist must be the same.",
                )
            )
        end
        return new{R,D}(λ, mark_dist)
    end
end

# Constructors
function PoissonProcess(λ::Vector{R}, mark_dist::D; check_args::Bool=true) where {R<:Real,D}
    return PoissonProcess(λ, fill(mark_dist, length(λ)); check_args=check_args)
end

function PoissonProcess(λ::Vector{R}; check_args::Bool=true) where {R<:Real}
    return PoissonProcess(λ, fill(Dirac(nothing), length(λ)); check_args=check_args)
end

function PoissonProcess(λ::R, mark_dist::D; check_args::Bool=true) where {R<:Real,D}
    return PoissonProcess([λ], [mark_dist]; check_args=check_args)
end

function PoissonProcess(λ::R; check_args::Bool=true) where {R<:Real}
    return PoissonProcess([λ], [Dirac(nothing)]; check_args=check_args)
end

PoissonProcess() = PoissonProcess(1.0)

function Base.show(io::IO, pp::PoissonProcess)
    if ndims(pp) == 1
        return print(io, "Univariate PoissonProcess($(pp.λ[1]), $(pp.mark_dist[1]))")
    else
        return print(io, "$(ndims(pp))-dimensional PoissonProcess($(pp.λ), $(pp.mark_dist))")
    end
end

## Alias 
"""
    UnmarkedPoissonProcess{R}

Homogeneous temporal Poisson process with scalar marginal intensities λ and no marks.

`UnmarkedPoissonProcess{R}` is simply a type alias for `PoissonProcess{R,Dirac{Nothing}}`.
"""
const UnmarkedPoissonProcess{R<:Real} = PoissonProcess{R,Dirac{Nothing}}

function Base.show(io::IO, pp::UnmarkedPoissonProcess)
    if ndims(pp) == 1
        return print(io, "Univariate Unmarked PoissonProcess($(pp.λ[1]))")
    else
        return print(io, "$(ndims(pp))-dimensional UnmarkedPoissonProcess($(pp.λ))")
    end
end

## Access

Base.ndims(pp::PoissonProcess) = length(pp.λ)

## AbstractPointProcess interface

mark_distribution(pp::PoissonProcess, t) = pp.mark_dist # For simulate_ogata
mark_distribution(pp::PoissonProcess, t, h) = pp.mark_dist
mark_distribution(pp::PoissonProcess, t, h, d) = pp.mark_dist[d]

ground_intensity(pp::PoissonProcess, t, h) = pp.λ
ground_intensity(pp::PoissonProcess, t, h, d) = pp.λ[d]

function intensity(pp::PoissonProcess, m, t, h)
    return pp.λ .* densityof.(mark_distribution(pp, t, h), m)
end

function intensity(pp::PoissonProcess, m, t, h, d)
    return pp.λ[d] .* densityof.(mark_distribution(pp, t, h, d), m)
end

function log_intensity(pp::PoissonProcess, m, t, h)
    return log.(ground_intensity(pp, t, h)) .+ logdensityof.(mark_distribution(pp, t, h), m)
end

function log_intensity(pp::PoissonProcess, m, t, h, d)
    return log(ground_intensity(pp, t, h, d)) + logdensityof(mark_distribution(pp, t, h, d), m)
end

function ground_intensity_bound(pp::PoissonProcess, t::T, h) where {T<:Real}
    B = ground_intensity(pp, t, h)
    L = typemax(T)
    return [(B[d], L) for d in 1:length(B)]
end

function ground_intensity_bound(pp::PoissonProcess, t::T, h, d) where {T<:Real}
    B = ground_intensity(pp, t, h, d)
    L = typemax(T)
    return (B, L)
end

function integrated_ground_intensity(pp::PoissonProcess, h::History, a, b)
    return ground_intensity(pp, a, h) .* (b - a)
end

function integrated_ground_intensity(pp::PoissonProcess, h::History, a, b, d)
    return ground_intensity(pp, a, h, d) * (b - a)
end

## Time change
function time_change(h::History, pp::PoissonProcess)
    times = [(event_times(h, d) .- min_time(h)) * pp.λ[d] for d in 1:ndims(h)]
    tmax = (h.tmax - h.tmin) * maximum(pp.λ)
    marks = [event_marks(h, d) for d in 1:ndims(h)]
    return History(times, 0, tmax, marks)
end
