"""
    MultivariatePoissonProcessPrior{R1,R2}

Gamma prior on all the event rates of a `MultivariatePoissonProcess`.

# Fields

- `α::Vector{R1}`
- `β::R2`
"""
struct PoissonProcessPrior{R1<:Real,R2<:Real}
    α::Vector{R1}
    β::R2
end

function DensityInterface.logdensityof(
    prior::PoissonProcessPrior, pp::PoissonProcess
)
    ground_int = pp.λ
    dim_int = ground_int ./ sum(ground_int)
    l = sum(
        logdensityof(Gamma(prior.α[d], inv(prior.β); check_args=false), dim_int[d])
        for d in 1:ndims(pp)
    )
    return l
end
