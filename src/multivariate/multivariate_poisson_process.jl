function PoissonProcess(λ::Vector{R}, mark_dists::Vector{D}; check_args::Bool=true) where {R<:Real,D}
    return IndependentMultivariateProcess(
        [PoissonProcess(λ[d], mark_dists[d]; check_args=check_args) for d in eachindex(λ)]
    )
end


function PoissonProcess(λ::Vector{R}) where {R<:Real}
    return PoissonProcess(λ, [Dirac(nothing) for _ in eachindex(λ)])
end


function PoissonProcess(λ::Vector{R}, mark_dist::D; check_args::Bool=true) where {R<:Real,D}
    return PoissonProcess(λ, [mark_dist for _ in eachindex(λ)]; check_args=check_args)
end