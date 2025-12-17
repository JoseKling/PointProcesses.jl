"""
    MultivariateHawkesProcess{R} <: HawkesProcess

Unmarked multivariate temporal Hawkes process with exponential decay

For a process with m = 1, 2, ..., M marginal processes, the conditional intensity
function of the m-th process is given by

    λₘ(t) = μₘ + ∑_{l=1,...,M} ∑_{tˡᵢ < t} αₘₗ exp(-βₘ(t - tˡᵢ)),

where tˡᵢ is the i-th element in the l-th marginal process.
μ and ω are vectors of length M with elements μₘ, ωₘ. α is a M×M
matrix with elements αₘₗ.

The process is represented as a marked process, where each marginal
process m = 1, 2, ..., M is corresponds to events (tᵐᵢ, m), tᵐᵢ being
the i-th element with mark m.

This is the 'Linear Multidimensional EHP' in Section 2 of
[Bonnet, Dion-Blanc and Perrin (2024)](https://arxiv.org/pdf/2410.05008v1)

# Fields
- `μ::Vector{<:Real}`: baseline intensity (immigration rate)
- `α::Matrix{<:Real}`: Jump size.
- `ω::Vector{<:Real}`: Decay rate.
"""
struct MultivariateHawkesProcess{T} <: HawkesProcess{T,Categorical{Float64,Vector{Float64}}}
    μ::T
    α::Matrix{T}
    ω::Vector{T}
    mark_dist::Categorical{Float64,Vector{Float64}} # To keep consistent with PoissonProcess. Helps in simulation.
end

function Base.show(io::IO, hp::MultivariateHawkesProcess{T}) where {T<:Real}
    return print(
        io,
        "MultivariateHawkesProcess\nμ = $(T.(hp.μ .* probs(hp.mark_dist)))\nα = $(hp.α)\nω = $(hp.ω)",
    )
end

function HawkesProcess(
    μ::Vector{<:Real}, α::Matrix{<:Real}, ω::Vector{<:Real}; check_args::Bool=true
)
    check_args && check_args_Hawkes(μ, α, ω)
    R = promote_type(eltype(μ), eltype(α), eltype(ω))
    return MultivariateHawkesProcess(
        R(sum(μ)), R.(α), R.(ω), Categorical(Float64.(μ / sum(μ)))
    )
end

# For M independent marginal processes
function HawkesProcess(
    μ::Vector{<:Real}, α::Vector{<:Real}, ω::Vector{<:Real}; check_args::Bool=true
)
    return HawkesProcess(μ, diagm(α), ω; check_args=check_args)
end

function check_args_Hawkes(μ::Vector{<:Real}, α::Matrix{<:Real}, ω::Vector{<:Real})
    if (length(μ) != size(α)[2])
        throw(DimensionMismatch("α must have size $(length(μ))×$(length(μ))"))
    end
    if (length(μ) != length(ω))
        throw(DimensionMismatch("α and ω must have the same length"))
    end
    if any(α ./ ω' .>= 1)
        @warn """HawkesProcess: There are parameters αᵢⱼ and ωⱼ with
        αᵢⱼ / ωⱼ >= 1. This may cause problems, especially in simulations."""
    end
    if any(ω .<= zero(ω))
        throw(
            DomainError(
                "ω = $ω", "HawkesProcess: All elements of ω must be strictly positive."
            ),
        )
    end
    if any(μ .< zero(μ)) || any(α .< zero(α))
        throw(
            DomainError(
                "μ = $μ, α = $α",
                "HawkesProcess: All elements of μ, α and ω must be non-negative.",
            ),
        )
    end
    return nothing
end
