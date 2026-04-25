"""
    MultivariatePoissonProcess

This is a multivariate point process where each dimension is an independent univariate Poisson process.

Alias for `IndependentMultivariateProcess{<:PoissonProcess}`.
"""
const MultivariatePoissonProcess = IndependentMultivariateProcess{<:PoissonProcess}

function PoissonProcess(
    λ::AbstractVector{R}, mark_dists::Vector{D}; check_args::Bool=true
) where {R<:Real,D}
    return IndependentMultivariateProcess([
        PoissonProcess(λ[d], mark_dists[d]; check_args=check_args) for d in eachindex(λ)
    ])
end

function PoissonProcess(λ::AbstractVector{R}; check_args::Bool=true) where {R<:Real}
    return PoissonProcess(λ, [Dirac(nothing) for _ in eachindex(λ)]; check_args=check_args)
end

function PoissonProcess(
    λ::AbstractVector{R}, mark_dist::D; check_args::Bool=true
) where {R<:Real,D}
    return PoissonProcess(λ, [mark_dist for _ in eachindex(λ)]; check_args=check_args)
end

function Base.show(io::IO, pp::MultivariatePoissonProcess)
    return print(
        io,
        "MultivariatePoissonProcess($([pp.processes[d].λ for d in 1:ndims(pp)]), $([typeof(pp.processes[d].mark_dist) for d in 1:ndims(pp)]))",
    )
end

"""
    MultivariatePoissonProcessPrior{R1,R2}

Gamma prior on all the event rates of a `MultivariatePoissonProcess`.

# Fields

- `α::Vector{R1}`
- `β::R2`
"""
struct MultivariatePoissonProcessPrior{R1<:Real,R2<:Real}
    α::Vector{R1}
    β::R2
end

function DensityInterface.logdensityof(
    prior::MultivariatePoissonProcessPrior, pp::MultivariatePoissonProcess
)
    λ = sum(pp.processes[d].λ for d in 1:ndims(pp))
    l = sum(
        logdensityof(
            Gamma(prior.α[d], inv(prior.β); check_args=false), pp.processes[d].λ / λ
        ) for d in 1:ndims(pp)
    )
    return l
end
