"""
    StatsAPI.fit(::Type{<:UnmarkedUnivariateHawkesProcess{R}}, h::History; step_tol::Float64=1e-6, max_iter::Int=1000, rng::AbstractRNG=default_rng()) where {R<:Real}

Expectation-Maximization algorithm from [E. Lewis, G. Mohler (2011)](https://arxiv.org/pdf/1801.08273)).
The relevant calculations are in page 4, equations 6-13.

Let (t₁ < ... < tₙ) be the event times over the interval [0, T). We use the immigrant-descendant representation,
where immigrants arrive at a constant base rate μ and each each arrival may generate descendants following the
activation function α exp(-ω(t - tᵢ)).

The algorithm consists in the following steps:
1. Start with some initial guess for the parameters μ, ψ, and ω. ψ = α ω is the branching factor.
2. Calculate λ(tᵢ; μ, ψ, ω) (`lambda_ts` in the code) using the procedure in [Ozaki (1979)](https://doi.org/10.1007/bf02480272).
3. For each tᵢ and each j < i, calculate Dᵢⱼ = P(tᵢ is a descendant of tⱼ) as

    Dᵢⱼ = ψ ω exp(-ω(tᵢ - tⱼ)) / λ(tᵢ; μ, ψ, ω).

    Define D = ∑_{j < i} Dᵢⱼ (expected number of descendants) and div = ∑_{j < i} (tᵢ - tⱼ) Dᵢⱼ. 
4. Update the parameters as
        μ = (N - D) / T
        ψ = D / N
        ω = D / div
5. If convergence criterion is met, return updated parameters, otherwise, back to step 2.

Notice that, for numerical stability, the process is normalized so the average inter-event time is equal to 1 and, 
therefore, the interval of the process is transformed from T to N. Also, in equation (8) in the paper,

∑_{i=1:n} pᵢᵢ = ∑_{i=1:n} (1 - ∑_{j < i} Dᵢⱼ) = N - D.
"""
function StatsAPI.fit(
    ::Type{<:UnmarkedUnivariateHawkesProcess{R}},
    h::History;
    step_tol::Float64=1e-6,
    max_iter::Int=1000,
    rng::AbstractRNG=default_rng(),
) where {R<:Real}
    unmarked_h = History(; times=h.times, tmin=h.tmin, tmax=h.tmax)
    params = fit_hawkes_params(
        R, unmarked_h, nb_events(h); step_tol=step_tol, max_iter=max_iter, rng=rng
    )
    return HawkesProcess(params...)
end

function StatsAPI.fit(
    ::Type{<:UnmarkedUnivariateHawkesProcess{R}},
    h::History{R2, Nothing};
    step_tol::Float64=1e-6,
    max_iter::Int=1000,
    rng::AbstractRNG=default_rng(),
) where {R<:Real, R2<:Real}
    params = fit_hawkes_params(
        R, h, nb_events(h); step_tol=step_tol, max_iter=max_iter, rng=rng
    )
    return HawkesProcess(params...)
end