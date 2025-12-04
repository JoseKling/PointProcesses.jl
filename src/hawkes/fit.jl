#=
Method for fitting the parameters of marked and unmarked Hawkes
processes. For unmarked processes, the marks in `history` must
be either all equal to `nothing` or equal to 1.
=#
function fit_hawkes_params(
    ::Type{R},
    h::History,
    div_ψ::Real;
    step_tol::Float64=1e-6,
    max_iter::Int=1000,
    rng::AbstractRNG=default_rng(),
) where {R<:Real}
    T = float(R)
    div_ψ = T(div_ψ)
    n = nb_events(h)
    nT = T(n)
    n == 0 && return (zero(T), zero(T), zero(T))

    tmax = T(duration(h))
    # Normalize times so average inter-event time is 1 (T -> n)
    norm_ts = T.(h.times .* (n / tmax))

    # preallocate
    A = zeros(T, n)            # A[i] = sum_{j<i} mⱼ exp(-ω (t_i - t_j))
    S = zeros(T, n)            # S[i] = sum_{j<i} mⱼ (t_i - t_j) exp(-ω (t_i - t_j))
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
            A[i] = update_A(A[i - 1], e, h.marks[i - 1]) # Different updates for real marks and no marks
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
        new_μ = one(T) - (D / nT)
        new_ψ = D / div_ψ
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
    return (μ * (nT / tmax), ψ * ω * (nT / tmax), ω * (nT / tmax))
end
