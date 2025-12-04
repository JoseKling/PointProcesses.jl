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

mark_distribution(hp::UnivariateHawkesProcess, _...) = hp.mark_dist

function HawkesProcess(μ::Real, α::Real, ω::Real, mark_dist; check_args::Bool=true)
    check_args && check_args_Hawkes(μ, α, ω, mark_dist)
    return UnivariateHawkesProcess(promote(μ, α, ω)..., mark_dist)
end

function check_args_Hawkes(μ::Real, α::Real, ω::Real, mark_dist)
    mean_α = mark_dist isa Dirac{Nothing} ? α : mean(mark_dist) * α
    check_args_Hawkes(μ, mean_α, ω)
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
