"""
    HomogeneousPoissonProcess{T<:Real}

Univariate Temporal Poisson process

# Fields
- `μ::T`: event rate
"""
struct HomogeneousPoissonProcess{T<:Real} <: AbstractPoissonProcess
    λ::T

    function HomogeneousPoissonProcess(λ::Real)
        if λ < 0
            throw(DomainError(λ, "Event rate λ must be positive."))
        end
        new{typeof(λ)}(λ)
    end
end

function simulate(rng::AbstractRNG, pp::HomogeneousPoissonProcess, tmin::Real, tmax::Real)
    times = simulate_poisson_times(rng, pp.λ, tmin, tmax)
    return History(; times=times, tmin=tmin, tmax=tmax, check=false)
end

function simulate(pp::HomogeneousPoissonProcess, tmin::Real, tmax::Real)
    return simulate(default_rng(), pp, tmin, tmax)
end
"""
    StatsAPI.fit(rng, ::Type{HomogeneousPoissonProcess}, h::History)

Estimates the parameters of a homogeneous Poisson process from event history `h`
"""
function StatsAPI.fit(::Type{HomogeneousPoissonProcess{T}}, h::History) where {T<:Real}
    return HomogeneousPoissonProcess(T(nb_events(h) / duration(h)))
end

# Type parameter for `HomogeneousPoissonProcess` was NOT explicitly provided
function StatsAPI.fit(
    HP::Type{HomogeneousPoissonProcess}, h::History{H,M}
) where {H<:Real,M}
    T = promote_type(Float64, H)
    return fit(HP{T}, h)
end

function time_change(pp::HomogeneousPoissonProcess, h::History)
    times = (h.times .- h.tmin) * pp.λ
    tmax = duration(h) * pp.λ
    return History(; times=times, marks=h.marks, tmin=0, tmax=tmax, check=false)
end

ground_intensity(pp::HomogeneousPoissonProcess) = pp.λ
intensity(pp::HomogeneousPoissonProcess) = ground_intensity(pp)
