struct HawkesProcess{T<:Real} <: AbstractPointProcess
    μ::T
    α::T
    ω::T

    function HawkesProcess(μ::T1, α::T2, ω::T3) where {T1,T2,T3}
        any((μ, α, ω) .< 0) &&
            throw(DomainError((μ, α, ω), "All parameters must be non-negative."))
        T = promote_type(T1, T2, T3)
        (μ_T, α_T, ω_T) = convert.(T, (μ, α, ω))
        new{T}(μ_T, α_T, ω_T)
    end
end

function Base.rand(rng::AbstractRNG, hp::HawkesProcess, tmin, tmax)
    sim = simulate_poisson_times(rng, hp.μ, tmin, tmax) # Simulate Poisson process with base rate
    sim_desc = generate_descendants(rng, sim, tmax, hp.α, hp.ω) # Recursively generates descendants from first events
    append!(sim, sim_desc)
    sort!(sim)
    return History(sim, fill(nothing, length(sim)), tmin, tmax)
end

"""
    StatsAPI.fit(rng, ::HawkesProcess, h::History; step_tol::Float64 = 1e-3, max_iter::Int = 1000)

Expectation-Maximization algorithm from [E. Lewis, G. Mohler (2011)](https://api.semanticscholar.org/CorpusID:2105771)).
The relevant calculations are in page 4, equations 6-13.

Let (t_1 < ... < t_N) be the event times over the interval [0, T). We use the immigrant-descendant representation,
where immigrants arrive at a constant base rate μ and each each arrival may generate descendants following the
activation function α exp(-ω(t - t_i)).

The algorithm consists in the following steps:
1. Start with some initial guess for the parameters μ, ψ, and ω. ψ = α ω is the branching factor.
2. Calculate λ(t_i; μ, ψ, ω) (`lambda_ts` in the code) using the procedure in [Ozaki (1979)](https://doi.org/10.1007/bf02480272).
3. For each t_i and each j < i, calculate D_{ij} = P(t_i is a descendant of t_j).
        D_{ij} = ψ ω exp(-ω(t_i - t_j)) / λ(t_i; μ, ψ, ω)
    Define D = sum_{j < i} D_{ij} (expected number of descendants) and div = sum_{j < i} (t_i - t_j) D_ij. 
4. Update the parameters as
        μ = (N - D) / T
        ψ = D / N
        ω = D / div
5. If convergence criterion is met, return updated parameters, otherwise, back to step 2.

Notice that in the implementation the process is normalized so the average inter-event time is equal to 1 and, 
therefore, the interval of the process is transformed from T to N. Also, in equation (8) in the paper,
sum_{i=1}^N p_{ii} = sum_{i=1}^N (1 - sum{j<i} D_{ij}) = N - D.
"""
function StatsAPI.fit(
    rng::AbstractRNG,
    ::Type{HawkesProcess},
    h::History;
    step_tol::Float64=1e-4,
    max_iter::Int=1000,
)
    N = nb_events(h)
    N == 0 && return HawkesProcess(0.0, 0.0, 0.0)
    T = duration(h)
    norm_ts = h.times .* (N / T) # Average inter-event time equal to 1. Numerical stability
    n_iters = 0
    step = step_tol + 1.0
    lambda_ts = zeros(N)
    # Step 1 - Choose initial guess such that Λ(T) ≈ N. After normalization, T → N
    μ = 0.2 + (0.6 * rand(rng)) # μ should not be too close to 0 or 1. μ = 1 → the base rate already accounts for all events
    ψ = (1 - μ) # ψ = α / β is the integral of the activation function. Λ(t) ≈ μt + ψN_t
    t90 = 0.5 + (1.5 * rand(rng)) # t90 is the time it takes for the activation function to decay by 90%. Set to t90 ∈ [1/2, 2]
    ω = log(10.0) * t90 # ω is the parameter corresponding to t90.
    while (step >= step_tol) && (n_iters < max_iter)
        # Step 2
        lambda_ts[1] = 0.0
        @inbounds for i in 2:N
            lambda_ts[i] =
                exp(-ω * (norm_ts[i] - norm_ts[i - 1])) * (1.0 + lambda_ts[i - 1])
        end
        lambda_ts .*= (ψ * ω)
        lambda_ts .+= μ
        # Step 3
        D = 0.0 # Expected number of descendants
        div = 0.0 # Needed to calculate the new parameters for the next iteration
        @inbounds for i in 2:N
            @inbounds for j in 1:(i - 1)
                diffs = norm_ts[i] - norm_ts[j]
                D_ij = (ψ * ω * exp(-ω * diffs)) / lambda_ts[i] # Probability that t_i is a descendant of t_j
                D += D_ij
                div += (diffs * D_ij)
            end
        end
        # Steps 4 and 5
        step = max(abs(μ - (1 - (D / N))), abs(ψ - (D / N)), abs(ω - (D / div)))
        n_iters += 1
        μ, ψ, ω = 1 - (D / N), D / N, D / div # Update parameters for the next iteration
    end
    n_iters >= max_iter &&
        @warn("Maximum number of iterations reached without convergence.")
    return HawkesProcess(μ * (N / T), ψ * ω * (N / T), ω * (N / T)) # Unnormalize parameters
end

function StatsAPI.fit(
    HP::Type{HawkesProcess}, h::History; step_tol::Float64=1e-3, max_iter::Int=1000
)
    fit(default_rng(), HP, h; step_tol=step_tol, max_iter=max_iter)
end

function time_change(hp::HawkesProcess, h::History)
    N = nb_events(h)
    A = zeros(N + 1) # Array A in Ozaki (1979)
    @inbounds for i in 2:N
        A[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + A[i - 1])
    end
    A[end] = exp(-hp.ω * (h.tmax - h.times[end])) * (1 + A[end - 1]) # Used to calculate the integral of the intensity at every event time
    times = hp.μ .* (h.times .- h.tmin) # Transformation with respect to base rate
    T_base = hp.μ * duration(h) # Contribution of base rate to total length of time re-scaled process
    @inbounds for i in eachindex(times)
        times[i] += (hp.α / hp.ω) * ((i - 1) - A[i]) # Add contribution of activation functions
    end
    return History(times, h.marks, 0.0, T_base + ((hp.α / hp.ω) * (N - A[end]))) # A time re-scaled process starts at t=0
end

function ground_intensity(hp::HawkesProcess, h::History, t)
    activation = sum(exp.(hp.ω .* (@view h.times[1:(searchsortedfirst(h.times, t) - 1)])))
    return hp.μ + (hp.α * activation / exp(hp.ω * t))
end

function integrated_ground_intensity(hp::HawkesProcess, h::History, tmin, tmax)
    times = event_times(h, h.tmin, tmax)
    integral = 0.0
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

# Internal function for simulating Hawkes processes
# The first generation, gen_0, is the `immigrants`, which is a set of event times.
# For each t_g ∈ gen_n, simulate an inhomogeneous Poisson process over the interval [t_g, T]
# with intensity λ(t) = α exp(-ω(t - t_g)) with the inverse method.
# gen_{n+1} is the set of all events simulated from all events in gen_n.
# The algorithm stops when the simulation from one generation results in no further events.
function generate_descendants(rng::AbstractRNG, immigrants::Vector, T, α, ω)
    descendants = eltype(immigrants)[]
    next_gen = immigrants
    while ~isempty(next_gen)
        # OPTIMIZE: Can this be improved by avoiding allocations of `curr_gen` and `next_gen`? Or does the compiler take care of that?
        curr_gen = copy(next_gen) # The current generation from which we simulate the next one
        next_gen = eltype(immigrants)[] # Gathers all the descendants from the current generation
        for parent in curr_gen # Generate the descendants for each individual event with the inverse method
            activation_integral = (α / ω) * (1.0 - exp(ω * (parent - T)))
            sim_transf = simulate_poisson_times(rng, 1.0, 0.0, activation_integral)
            @. sim_transf = parent - ((1.0 / ω) * log(1.0 - ((ω / α) * sim_transf))) # Inverse of integral of the activation function
            append!(next_gen, sim_transf)
        end
        append!(descendants, next_gen)
    end
    return descendants
end
