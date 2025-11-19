"""
    HawkesProcess <: AbstractPointProcess

Common interface for all subtypes of `HawkesProcess`.
"""
abstract type HawkesProcess <: AbstractPointProcess end

"""
    UnivariateHawkesProcess{R<:Real,D} <: HawkesProcess

Univariate Hawkes process with exponential decay kernel and mark 
distribution `D`.

Denote the events of the process by (tᵢ, mᵢ), where tᵢ is the event time
and mᵢ ∈ M the corresponding mark. The conditional intensity function
of the Hawkes process is given by

λ(t) = μ + ∑_{tᵢ < t} α mᵢ exp(-ω (t - tᵢ)).

Notice that the mark only affects the jump size.

# Fields
- `μ::R`: baseline intensity (immigration rate)
- `α::R`: jump size
- `ω::R`: decay rate.
- `mark_dist::D`: distribution of marks
"""
struct UnivariateHawkesProcess{T<:Real,D} <: HawkesProcess
    μ::T
    α::T
    ω::T
    mark_dist::D
end

function Base.show(io::IO, hp::UnivariateHawkesProcess)
    return print(io, "UnivariateHawkesProcess($(hp.μ), $(hp.α), $(hp.ω), $(hp.mark_dist))")
end

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

Alias for UnivariateHawkesProcess{R, Dirac{Nothing}}.
"""
const UnmarkedUnivariateHawkesProcess{R<:Real} = UnivariateHawkesProcess{R,Dirac{Nothing}}

function Base.show(io::IO, hp::UnmarkedUnivariateHawkesProcess)
    return print(io, "UnmarkedUnivariateHawkesProcess$((hp.μ, hp.α, hp.ω))")
end

"""
    MultivariateHawkesProcess{R} <: HawkesProcess

Unmarked multivariate temporal Hawkes process with exponential decay

For a process with m = 1, 2, ..., M marginal processes, the conditional intensity
function of the m-th process is given by

    λₘ(t) = μₘ + ∑_{l=1,...,M} ∑_{tˡᵢ < t} αₘₗ exp(-βₘₗ(t - tˡᵢ)),

where tˡᵢ is the i-th element in the l-th marginal process.
μ is a vector of length M with elements μₘ. α and ω are M×M
matrices with elements αₘₗ and ωₘₗ.

The process is represented as a marked process, where each marginal
process m = 1, 2, ..., M is represented by events (tᵐᵢ, m), tᵐᵢ being
the i-th element with mark m.

# Fields
- `μ::Vector{<:Real}`: baseline intensity (immigration rate)
- `α::Matrix{<:Real}`: Jump size.
- `ω::Matrix{<:Real}`: Decay rate.
"""
struct MultivariateHawkesProcess{T<:Real} <: HawkesProcess
    μ::T
    α::Matrix{T}
    ω::Matrix{T}
    mark_dist::Categorical{Float64,Vector{Float64}} # To keep consistent with PoissonProcess. Helps in simulation.
end

function Base.show(io::IO, hp::MultivariateHawkesProcess{T}) where {T<:Real}
    return print(
        io,
        "MultivariateHawkesProcess\nμ = $(T.(hp.μ .* probs(hp.mark_dist)))\nα = $(hp.α)\nω = $(hp.ω)",
    )
end

Base.length(mh::MultivariateHawkesProcess) = size(mh.α)[1]

## Constructors
### UnivariatehawkesProcess
function HawkesProcess(μ::Real, α::Real, ω::Real, mark_dist; check_args::Bool=true)
    check_args && check_args_Hawkes(μ, α, ω, mark_dist)
    return UnivariateHawkesProcess(promote(μ, α, ω)..., mark_dist)
end

function HawkesProcess(μ::Real, α::Real, ω::Real; check_args::Bool=true)
    return HawkesProcess(μ, α, ω, Dirac(nothing); check_args=check_args)
end

### MultivariateHawkesProcess
function HawkesProcess(
    μ::Vector{<:Real}, α::Matrix{<:Real}, ω::Matrix{<:Real}; check_args::Bool=true
)
    check_args && check_args_Hawkes(μ, α, ω)
    R = promote_type(eltype(μ), eltype(α), eltype(ω))
    return MultivariateHawkesProcess(R(sum(μ)), R.(α), R.(ω), Categorical(μ / sum(μ)))
end

# For M independent marginal processes
function HawkesProcess(
    μ::Vector{<:Real}, α::Vector{<:Real}, ω::Vector{<:Real}; check_args::Bool=true
)
    check_args && check_args_Hawkes(μ, diagm(α), diagm(ω))
    R = promote_type(eltype(μ), eltype(α), eltype(ω))
    return MultivariateHawkesProcess(R(sum(μ)), R.(α), R.(ω), Categorical(μ / sum(μ)))
end

# Check args
## Univariate Hawkes
function check_args_Hawkes(μ::Real, α::Real, ω::Real, mark_dist)
    if any((μ, α, ω) .< 0)
        throw(
            DomainError(
                "μ = $μ, α = $α, ω = $ω",
                "HawkesProcess: All parameters must be non-negative.",
            ),
        )
    end
    mean_α = mark_dist isa Dirac{Nothing} ? α : mean(mark_dist) * α
    if mean_α >= ω
        throw(
            DomainError(
                "α = $(mean_α), ω = $ω",
                "HawkesProcess: mᵢα must be, on average, smaller than ω. Stability condition.",
            ),
        )
    end
    if !isa(mark_dist, Dirac{Nothing}) && minimum(mark_dist) < 0
        throw(
            DomainError(
                "Mark distribution support = $((support(mark_dist).lb, support(mark_dist).ub))",
                "HawkesProcess: Support of mark distribution must be contained in non-negative numbers",
            ),
        )
    end
    return nothing
end

## Multivariate Hawkes
function check_args_Hawkes(μ::Vector{<:Real}, α::Matrix{<:Real}, ω::Matrix{<:Real})
    if length(μ) != size(α)[2]
        throw(DimansionMismatch("α must have size $(length(μ))×$(length(μ))"))
    end
    if length(μ) != size(ω)[2]
        throw(DimansionMismatch("ω must have size $(length(μ))×$(length(μ))"))
    end
    if any(sum(α ./ ω; dims=1) .>= 1)
        throw(
            DomainError(
                "α = $α, ω = $ω",
                "HawkesProcess: Sum of α/β over each row must be smaller than 1. Stability condition.",
            ),
        )
    end
    if any(μ .< zero(μ)) || any(α .< zero(α)) || any(ω .< zero(ω))
        throw(
            DomainError(
                "μ = $μ, α = $α, ω = $ω",
                "HawkesProcess: All elements of μ, α and ω must be non-negative.",
            ),
        )
    end
    ω[α .== 0] .= 1 # Protects against division by 0 in simulation
    return nothing
end
