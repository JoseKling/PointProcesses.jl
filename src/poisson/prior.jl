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
    prior::MultivariatePoissonProcessPrior, pp::PoissonProcess
)
    l = sum(
        logdensityof(Gamma(prior.α[m], inv(prior.β); check_args=false), intensity(pp, m))
        for m in 1:length(pp)
    )
    return l
end
