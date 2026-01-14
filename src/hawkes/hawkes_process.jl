"""
    HawkesProcess{T<:Real}

Univariate Hawkes process with exponential decay kernel.

A Hawkes process is a self-exciting point process where each event increases the probability
of future events. The conditional intensity function is given by:

    λ(t) = μ + α ∑_{tᵢ < t} exp(-ω(t - tᵢ))

where the sum is over all previous event times tᵢ.

# Fields

- `μ::T`: baseline intensity (immigration rate)
- `α::T`: jump size (immediate increase in intensity after an event)  
- `ω::T`: decay rate (how quickly the excitement fades)

Conditions:
- μ, α, ω >= 0
- ψ = α/ω < 1 → Stability condition. ψ is the expected number of events each event generates

Following the notation from [Lewis2011](@cite).
"""
struct HawkesProcess{T<:Real} <: AbstractPointProcess
    μ::T
    α::T
    ω::T

    function HawkesProcess(μ::T1, α::T2, ω::T3) where {T1,T2,T3}
        any((μ, α, ω) .< 0) &&
            throw(DomainError((μ, α, ω), "All parameters must be non-negative."))
        (α > 0 && α >= ω) &&
            throw(DomainError((α, ω), "Parameter ω must be strictly smaller than α"))
        T = promote_type(T1, T2, T3)
        (μ_T, α_T, ω_T) = convert.(T, (μ, α, ω))
        new{T}(μ_T, α_T, ω_T)
    end
end

function simulate(rng::AbstractRNG, hp::HawkesProcess, tmin, tmax)
    sim = simulate_poisson_times(rng, hp.μ, tmin, tmax) # Simulate Poisson process with base rate
    sim_desc = generate_descendants(rng, sim, tmax, hp.α, hp.ω) # Recursively generates descendants from first events
    append!(sim, sim_desc)
    sort!(sim)
    return History(sim, tmin, tmax, check_args=false)
end

"""
    StatsAPI.fit(rng, ::Type{HawkesProcess{T}}, h::History; step_tol::Float64 = 1e-6, max_iter::Int = 1000) where {T<:Real}

Expectation-Maximization algorithm from [Lewis2011](@cite).
The relevant calculations are in page 4, equations 6-13.

Let (t₁ < ... < tₙ) be the event times over the interval [0, T). We use the immigrant-descendant representation,
where immigrants arrive at a constant base rate μ and each each arrival may generate descendants following the
activation function α exp(-ω(t - tᵢ)).

The algorithm consists in the following steps:
1. Start with some initial guess for the parameters μ, ψ, and ω. ψ = α ω is the branching factor.
2. Calculate λ(tᵢ; μ, ψ, ω) (`lambda_ts` in the code) using the procedure in [Ozaki1979](@cite).
3. For each tᵢ and each j < i, calculate Dᵢⱼ = P(tᵢ is a descendant of tⱼ) as

    Dᵢⱼ = ψ ω exp(-ω(tᵢ - tⱼ)) / λ(tᵢ; μ, ψ, ω).

    Define D = ∑_{j < i} Dᵢⱼ (expected number of descendants) and div = ∑_{j < i} (tᵢ - tⱼ) Dᵢⱼ. 
4. Update the parameters as
        μ = (N - D) / T
        ψ = D / N
        ω = D / div
5. If convergence criterion is met, return updated parameters, otherwise, back to step 2.

Notice that, in the implementation, the process is normalized so the average inter-event time is equal to 1 and, 
therefore, the interval of the process is transformed from T to N. Also, in equation (8) in the paper,

∑_{i=1:n} pᵢᵢ = ∑_{i=1:n} (1 - ∑_{j < i} Dᵢⱼ) = N - D.

"""
function StatsAPI.fit(
    ::Type{HawkesProcess{T}},
    h::UnivariateHistory;
    step_tol::Float64=1e-6,
    max_iter::Int=1000,
    rng::AbstractRNG=default_rng(),
) where {T<:Real}
    n = nb_events(h)
    n == 0 && return HawkesProcess(zero(T), zero(T), zero(T))

    tmax = T(duration(h))
    # Normalize times so average inter-event time is 1 (T -> n)
    norm_ts = T.(h.times .* (n / tmax))

    # preallocate
    A = zeros(T, n)            # A[i] = sum_{j<i} exp(-ω (t_i - t_j))
    S = zeros(T, n)            # S[i] = sum_{j<i} (t_i - t_j) exp(-ω (t_i - t_j))
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
            Ai_1 = A[i - 1]
            A[i] = e * (one(T) + Ai_1)
            S[i] = e * (S[i - 1] + Δ * (one(T) + Ai_1))
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
        new_μ = one(T) - (D / n)
        new_ψ = D / n
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
    return HawkesProcess(μ * (n / tmax), ψ * ω * (n / tmax), ω * (n / tmax))
end

# Type parameter for `HawkesProcess` was NOT explicitly provided
function StatsAPI.fit(HP::Type{<:HawkesProcess}, h::UnivariateHistory{H,M}; kwargs...) where {H<:Real,M}
    T = promote_type(Float64, H)
    return fit(HP{T}, h; kwargs...)
end

function time_change(h::UnivariateHistory{R,M}, hp::HawkesProcess) where {R<:Real,M}
    T = float(R)
    n = nb_events(h)
    n == 0 && return History(T[], zero(T), T(hp.μ * duration(h)), event_marks(h))
    times = zeros(T, n)

    # In this step, `times` is the vector A in Ozaki (1979)
    #     A[1] = 0, A[i] = exp(-ω(tᵢ - tᵢ₋₁)) (1 + A[i - 1])
    for i in 2:n
        times[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + times[i - 1])
    end
    tmax = exp(-hp.ω * (h.tmax - h.times[end])) * (1 + times[end])

    # Integral of the intensity corresponding to activation functions
    #    Λ(tₙ) - μtₙ  = (α/ω) ((n-1) - A[n])
    for i in eachindex(times)
        times[i] = (hp.α / hp.ω) * ((i - 1) - times[i]) # Add contribution of activation functions
    end
    tmax = (hp.α / hp.ω) * (n - tmax) # Add contribution of activation functions

    # Add integral of base intensity
    times .+= T.(hp.μ .* (h.times .- h.tmin))
    tmax += T(hp.μ * duration(h))

    return History(times, zero(T), tmax, event_marks(h); check_args=false) # A time re-scaled process starts at t=0
end

function ground_intensity(hp::HawkesProcess, h::History, t)
    activation = sum(exp.(hp.ω .* (@view h.times[1:(searchsortedfirst(h.times, t) - 1)])))
    return hp.μ + (hp.α * activation / exp(hp.ω * t))
end

function integrated_ground_intensity(hp::HawkesProcess, h::History, tmin, tmax)
    times = event_times(h, h.tmin, tmax)
    integral = 0
    for ti in times
        # Integral of activation function. 'max(tmin - ti, 0)' corrects for events that occurred
        # inside or outside the interval [tmin, tmax].
        integral += (exp(-hp.ω * max(tmin - ti, 0)) - exp(-hp.ω * (tmax - ti)))
    end
    integral *= hp.α / hp.ω
    integral += hp.μ * (tmax - tmin) # Integral of base rate
    return integral
end

function DensityInterface.logdensityof(hp::HawkesProcess, h::History)
    A = zeros(nb_events(h)) # Vector A in Ozaki (1979)
    for i in 2:nb_events(h)
        A[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + A[i - 1])
    end
    return sum(log.(hp.μ .+ (hp.α .* A))) - # Value of intensity at each event
           (hp.μ * duration(h)) - # Integral of base rate
           ((hp.α / hp.ω) * sum(1 .- exp.(-hp.ω .* (duration(h) .- h.times)))) # Integral of each kernel
end

#=
Internal function for simulating Hawkes processes
The first generation, gen_0, is the `immigrants`, which is a set of event times.
For each t_g ∈ gen_n, simulate an inhomogeneous Poisson process over the interval [t_g, T]
with intensity λ(t) = α exp(-ω(t - t_g)) with the inverse method.
gen_{n+1} is the set of all events simulated from all events in gen_n.
The algorithm stops when the simulation from one generation results in no further events.
=#
function generate_descendants(
    rng::AbstractRNG, immigrants::Vector{T}, tmax, α, ω
) where {T<:Real}
    descendants = T[]
    next_gen = immigrants
    while !isempty(next_gen)
        # OPTIMIZE: Can this be improved by avoiding allocations of `curr_gen` and `next_gen`? Or does the compiler take care of that?
        curr_gen = copy(next_gen) # The current generation from which we simulate the next one
        next_gen = eltype(immigrants)[] # Gathers all the descendants from the current generation
        for parent in curr_gen # Generate the descendants for each individual event with the inverse method
            activation_integral = (α / ω) * (one(T) - exp(ω * (parent - tmax)))
            sim_transf = simulate_poisson_times(rng, one(T), zero(T), activation_integral)
            @. sim_transf = parent - (inv(ω) * log(one(T) - ((ω / α) * sim_transf))) # Inverse of integral of the activation function
            append!(next_gen, sim_transf)
        end
        append!(descendants, next_gen)
    end
    return descendants
end
