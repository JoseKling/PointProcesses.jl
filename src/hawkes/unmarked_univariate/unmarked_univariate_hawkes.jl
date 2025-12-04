"""
    UnmarkedUnivariateHawkesProcess{R<:Real} <: HawkesProcess

Unmarked univariate Hawkes process with exponential decay kernel.

Denote the events of the process by tᵢ. The conditional intensity function
of the Hawkes process is given by

λ(t) = μ + ∑_{tᵢ < t} α exp(-ω (t - tᵢ)).

# Fields
- `μ::R`: baseline intensity (immigration rate)
- `α::R`: jump size
- `ω::R`: decay rate.
"""
struct UnmarkedUnivariateHawkesProcess{T<:Real} <: HawkesProcess
    μ::T
    α::T
    ω::T
end

function Base.show(io::IO, hp::UnmarkedUnivariateHawkesProcess)
    return print(io, "UnmarkedUnivariateHawkesProcess$((hp.μ, hp.α, hp.ω))")
end

mark_distribution(::UnmarkedUnivariateHawkesProcess, _...) = Dirac(nothing)

function HawkesProcess(μ::Real, α::Real, ω::Real; check_args::Bool=true)
    check_args && check_args_Hawkes(μ, α, ω)
    return UnmarkedUnivariateHawkesProcess(promote(μ, α, ω)...)
end

function check_args_Hawkes(μ::Real, α::Real, ω::Real)
    if any((μ, α, ω) .< 0)
        throw(
            DomainError(
                "μ = $μ, α = $α, ω = $ω",
                "HawkesProcess: All parameters must be non-negative.",
            ),
        )
    end
    if α >= ω
        throw(
            DomainError(
                "α = $α, ω = $ω",
                "HawkesProcess: mᵢα must be, on average, smaller than ω. Stability condition.",
            ),
        )
    end
end
