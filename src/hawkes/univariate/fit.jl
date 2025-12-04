"""
    StatsAPI.fit(::Type{<:UnivariateHawkesProcess{R}}, h::History; step_tol::Float64=1e-6, max_iter::Int=1000, rng::AbstractRNG=default_rng()) where {R<:Real}

Adaptation of the Expectation-Maximization algorithm from [E. Lewis, G. Mohler (2011)](https://arxiv.org/pdf/1801.08273))
for marked processes. The relevant calculations are in page 4, equations 6-13.

Let (t₁, m₁), (t₂, m₂), ..., (tₙ, mₙ) be a marked event history with  (t₁ < ... < tₙ) over the interval
[0, T). We use the immigrant-descendant representation, where immigrants arrive at a constant base rate
μ and each each arrival may generate descendants following the activation function α mᵢ exp(-ω(t - tᵢ)).

The algorithm consists in the following steps:
1. Start with some initial guess for the parameters μ, ψ, and ω. ψ = α ω is the branching factor.
2. Calculate λ(tᵢ; μ, ψ, ω) (`lambda_ts` in the code) using the procedure in [Ozaki (1979)](https://doi.org/10.1007/bf02480272).
3. For each tᵢ and each j < i, calculate Dᵢⱼ = P(tᵢ is a descendant of tⱼ) as

    Dᵢⱼ = ψ ω mⱼ exp(-ω(tᵢ - tⱼ)) / λ(tᵢ; μ, ψ, ω).

    Define D = ∑_{j < i} Dᵢⱼ (expected number of descendants) and div = ∑_{j < i} (tᵢ - tⱼ) Dᵢⱼ. 
4. Update the parameters as
        μ = (N - D) / T
        ψ = D / ∑ mᵢ
        ω = D / div
5. If convergence criterion is met, return updated parameters, otherwise, back to step 2.

Notice that, for numerical stability, the process is normalized so the average inter-event time is equal to 1 and, 
therefore, the interval of the process is transformed from T to N. Also, in equation (8) in the paper,

∑_{i=1:n} pᵢᵢ = ∑_{i=1:n} (1 - ∑_{j < i} Dᵢⱼ) = N - D.
"""
function StatsAPI.fit(
    HP::Type{<:UnivariateHawkesProcess{R,D}},
    h::History;
    step_tol::Float64=1e-6,
    max_iter::Int=1000,
    rng::AbstractRNG=default_rng(),
) where {R,D}
    sum_marks = sum(h.marks)
    params = fit_hawkes_params(R, h, sum_marks; step_tol=step_tol, max_iter=max_iter, rng=rng)
    d = fit(D, h.marks)
    return HawkesProcess(params..., d)
end
