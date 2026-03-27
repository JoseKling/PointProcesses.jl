## Fit MLE

#=
A separate `fit` method for unmarked Poisson processes is needed, because
`Distributions.jl` does not provide a `fit` method for the `Dirac` distribution
=#
function StatsAPI.fit(
    ::Type{UnmarkedPoissonProcess{R}}, ss::PoissonProcessStats{R1,R2}; kwargs...
) where {R<:Real,R1<:Real,R2<:Real}
    λ = convert.(R, ss.nb_events ./ ss.duration)
    return PoissonProcess(λ; check_args=false)
end

function StatsAPI.fit(
    ::Type{PoissonProcess{R,D}}, ss::PoissonProcessStats{R1,R2}; kwargs...
) where {R<:Real,D,R1<:Real,R2<:Real}
    λ = convert.(R, ss.nb_events ./ ss.duration)
    mark_dist = [fit(D, ss.marks[d], ss.weights[d]) for d in 1:length(ss.marks)]
    return PoissonProcess(λ, mark_dist)
end

function StatsAPI.fit(pptype::Type{PoissonProcess{R,D}}, args...; kwargs...) where {R,D}
    ss = suffstats(pptype, args...)
    return fit(pptype, ss)
end

## Bayesian fit (only for MultivariatePoissonProcess)

function fit_map(
    ::Type{UnmarkedPoissonProcess{R}},
    prior::PoissonProcessPrior,
    ss::PoissonProcessStats;
    kwargs...,
) where {R<:Real}
    (; α, β) = prior
    posterior_nb_events = [length(ss.marks[d]) for d in 1:length(α)] .+ α
    posterior_duration = ss.duration + β
    λ = convert.(R, posterior_nb_events ./ posterior_duration)
    return PoissonProcess(λ; check_args=false)
end

function fit_map(
    pptype::Type{UnmarkedPoissonProcess{R}},
    prior::PoissonProcessPrior,
    args...;
    kwargs...,
) where {R<:Real}
    ss = suffstats(pptype, args..., kwargs...)
    return fit_map(pptype, prior, ss)
end
