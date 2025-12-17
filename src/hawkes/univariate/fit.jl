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

Notice that, for numerical stability, the process is normalized so the average inter-event time is equal to 1 and, therefore, the interval of the process is transformed from T to N. Also, in equation (8) in the paper,

∑_{i=1:n} pᵢᵢ = ∑_{i=1:n} (1 - ∑_{j < i} Dᵢⱼ) = N - D.
"""
function StatsAPI.fit(
    HP::Type{UnivariateHawkesProcess{R,D}},
    h::History;
    step_tol::Float64=1e-6,
    max_iter::Int=1000,
    rng::AbstractRNG=default_rng(),
) where {R,D}
    div_ψ = D == Dirac{Nothing} ? nb_events(h) : sum(h.marks)
    params = fit_hawkes_params(HP, h, div_ψ; step_tol=step_tol, max_iter=max_iter, rng=rng)
    d = D == Dirac{Nothing} ? Dirac(nothing) : fit(D, h.marks)
    return HawkesProcess(params..., d)
end

#=
Method for fitting the parameters of marked and unmarked Hawkes
processes.
`div_ψ` must be passed as an argument. For unmarked processes, 
`div_ψ` is equal to the number of events in the evet history, for
marked processes, `div_ψ` is equal to the sum of all the marks.
=#
function fit_hawkes_params(
    ::Type{UnivariateHawkesProcess{R,Dist}},
    h::History,
    div_ψ::Real;
    step_tol::Float64=1e-6,
    max_iter::Int=1000,
    rng::AbstractRNG=default_rng(),
) where {R<:Real,Dist}
    T = float(R)
    div_ψ = T(div_ψ)
    n = nb_events(h)
    N = T(n)
    n == 0 && return (zero(T), zero(T), zero(T))

    tmax = T(duration(h))
    # Normalize times so average inter-event time is 1 (T -> n)
    norm_ts = T.(h.times .* (n / tmax))

    # preallocate
    A = zeros(T, n)            # A[i] = sum_{j<i} α mⱼ exp(-ω (t_i - t_j))
    S = zeros(T, n)            # S[i] = sum_{j<i} α mⱼ (t_i - t_j) exp(-ω (t_i - t_j))
    lambda_ts = similar(A)     # λ_i

    # Λ(T) ≈ n after normalization
    # μ not too close to 0 or 1 so that both base and offspring contribute
    μ = T(0.2) + (T(0.6) * rand(rng, T))
    ψ = one(T) - μ                           # ψ = α/ω (branching ratio)
    t90 = T(0.5) + (T(1.5) * rand(rng, T))   # inverse of the time to decay to 10% in normalized time
    ω = log(T(10)) * t90

    n_iters = 0
    step = step_tol + one(T)

    # First event: A[1]=0, S[1]=0, so λ_1 = μ
    lambda_ts[1] = μ

    while (step >= step_tol) && (n_iters < max_iter)
        # compute A, S, and λ
        for i in 2:n
            Δ = norm_ts[i] - norm_ts[i - 1]
            e = exp(-ω * Δ)
            mi_1 = process_mark(Dist, h.marks[i - 1])
            A[i] = e * (mi_1 + A[i - 1]) # Different updates for real marks and no marks
            S[i] = (Δ * A[i]) + (e * S[i - 1])
            lambda_ts[i] = μ + (ψ * ω) * A[i]
        end

        # E-step aggregates
        D = zero(T)   # expected number of descendants
        div = zero(T)   # ∑ (t_i - t_j) D_{ij}
        for i in 2:n
            w = (ψ * ω) / lambda_ts[i]  # factor common to all j<i
            D += w * A[i]
            div += w * S[i]
        end

        # Steps 4–5: M-step updates + convergence check
        new_μ = (N - D) / N # N - D = ∑ pᵢᵢ
        new_ψ = (D / div_ψ)
        new_ω = D / (div + eps(T))   # small guard to avoid div=0

        step = max(abs(μ - new_μ), abs(ψ - new_ψ), abs(ω - new_ω))
        μ, ψ, ω = new_μ, new_ψ, new_ω
        n_iters += 1

        # Prepare for next loop: reset λ₁ since A[1]=0 regardless of params
        lambda_ts[1] = μ
    end

    n_iters >= max_iter && @warn "Maximum number of iterations reached without convergence."

    # Unnormalize back to original time scale (T -> tmax):
    # parameters in normalized space (') relate to original by μ0=μ'*(n/tmax), ω0=ω'*(n/tmax), α0=(ψ'ω')*(n/tmax)
    return (μ * (N / tmax), ψ * ω * (N / tmax), ω * (N / tmax))
end
