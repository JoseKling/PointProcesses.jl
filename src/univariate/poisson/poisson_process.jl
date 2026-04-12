"""
    PoissonProcess{R,D}

Homogeneous temporal Poisson process with arbitrary mark distribution.

# Fields

- `λ::R`: ground intensity.
- `mark_dist::D`: mark distribution.

# Constructor

    PoissonProcess(λ, mark_dist)
"""
struct PoissonProcess{R<:Real,D} <: AbstractUnivariateProcess
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
    return PoissonProcess(λ, Dirac(nothing); check_args=check_args)
end

PoissonProcess() = PoissonProcess(1.0)

function Base.show(io::IO, pp::PoissonProcess)
    return print(io, "PoissonProcess($(pp.λ), $(typeof(pp.mark_dist)))")
end

## Access
ground_intensity(pp::PoissonProcess) = pp.λ
mark_distribution(pp::PoissonProcess) = pp.mark_dist

## Intensity functions
function intensity(pp::PoissonProcess, m)
    return ground_intensity(pp) * densityof(mark_distribution(pp), m)
end

function log_intensity(pp::PoissonProcess, m)
    return log(ground_intensity(pp)) + logdensityof(mark_distribution(pp), m)
end

## Time change
function time_change(h::History, pp::PoissonProcess)
    times = (h.times .- h.tmin) .* ground_intensity(pp)
    tmax = (h.tmax - h.tmin) * ground_intensity(pp)
    return History(; times=times, tmin=0, tmax=tmax, marks=h.marks)
end

## Implementing AbstractPointProcess interface

ground_intensity(pp::PoissonProcess, t, h) = ground_intensity(pp)
mark_distribution(pp::PoissonProcess, t, h) = mark_distribution(pp)
mark_distribution(pp::PoissonProcess, t) = mark_distribution(pp) # For simulate_ogata
intensity(pp::PoissonProcess, m, t, h) = intensity(pp, m)
log_intensity(pp::PoissonProcess, m, t, h) = log_intensity(pp, m)

function ground_intensity_bound(pp::PoissonProcess, t::T, h) where {T<:Real}
    B = ground_intensity(pp)
    L = typemax(T)
    return (B, L)
end

function integrated_ground_intensity(pp::PoissonProcess, h, a, b)
    return ground_intensity(pp) * (b - a)
end
