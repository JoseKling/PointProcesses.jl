"""
    PoissonProcess{R,D}

Homogeneous temporal Poisson process with arbitrary mark distribution.

# Fields

- `λ::R`: ground intensity.
- `mark_dist::D`: mark distribution.

# Constructor

    PoissonProcess(λ, mark_dist)
"""
struct PoissonProcess{R<:Real,D<:PointProcessMarkDistribution} <: AbstractUnivariateProcess
    λ::R
    mark_dist::D

    function PoissonProcess(λ::R, mark_dist::D; check_args::Bool=true) where {R<:Real,D}
        check_args &&
            λ < zero(λ) &&
            throw(
                DomainError(
                    "λ = $λ", "PoissonProcess: the ground intensity λ must be non negative."
                ),
            )
        return new{R,D}(λ, mark_dist)
    end
end

function PoissonProcess(λ::R; check_args::Bool=true) where {R<:Real}
    return PoissonProcess(λ, NoMarks(); check_args=check_args)
end

PoissonProcess() = PoissonProcess(1.0)

function Base.show(io::IO, pp::PoissonProcess)
    return print(io, "PoissonProcess($(pp.λ), $(typeof(pp.mark_dist)))")
end

## Access
ground_intensity(pp::PoissonProcess, t, h) = pp.λ

## Time change
function time_change(h::History, pp::PoissonProcess)
    times = (h.times .- h.tmin) .* pp.λ
    tmax = (h.tmax - h.tmin) * pp.λ
    return History(; times=times, tmin=0, tmax=tmax, marks=h.marks)
end

## Implementing AbstractPointProcess interface
function ground_intensity_bound(pp::PoissonProcess{R}, t::T, h) where {R,T<:Real}
    U = promote_type(R, T)
    B = U(pp.λ)
    L = typemax(U)
    return (B, L)
end

function integrated_ground_intensity(pp::PoissonProcess, h, a, b)
    return pp.λ * (b - a)
end
