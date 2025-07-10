struct HawkesProcess <: AbstractPointProcess
    μ::Float64
    α::Float64
    β::Float64
end

function Base.rand(rng::AbstractRNG, hp::HawkesProcess, tmin, tmax)
    sim = tmin .+ simulate_poisson(rng, hp.μ, tmax - tmin) # Simulate Poisson process with base rate
    sim_desc = generate_descendants(rng, sim, tmax, hp.α, hp.β) # Recursively generates descendants from first events
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
activation function α exp(-β(t - t_i)).

The algorithm consists in the following steps:
1. Start with some initial guess for the parameters μ, ψ, and β. ψ = α β is the branching factor.
2. Calculate λ(t_i; μ, ψ, β) (`lambda_ts` in the code) using the procedure in [Ozaki (1979)](https://doi.org/10.1007/bf02480272).
3. For each t_i and each j < i, calculate D_{ij} = P(t_i is a descendant of t_j).
        D_{ij} = ψ β exp(-β(t_i - t_j)) / λ(t_i; μ, ψ, β)
    Define D = sum_{j < i} D_{ij} (expected number of descendants) and div = sum_{j < i} (t_i - t_j) D_ij. 
4. Update the parameters as
        μ = (N - D) / T
        ψ = D / N
        β = D / div
5. If convergence criterion is met, return updated parameters, otherwise, back to step 2.

Notice that in the implementation the process is normalized so the average inter-event time is equal to 1 and, 
therefore, the interval of the process is transformed from T to N. Also, in equation (8) in the paper,
sum_{i=1}^N p_{ii} = sum_{i=1}^N (1 - sum{j<i} D_{ij}) = N - D.
"""
function StatsAPI.fit(
    rng::AbstractRNG,
    ::Type{HawkesProcess},
    h::History;
    step_tol::Float64=1e-3,
    max_iter::Int=1000,
)
    N = nb_events(h)
    N == 0 && return HawkesProcess(0.0, 0.0, 0.0)
    T = duration(h)
    c1 = 0.2 + (0.6 * rand(rng))
    c2 = 0.1 + (0.8 * rand(rng))
    norm_ts = h.times .* (N / T) # Average inter-event time equal to 1. Numerical stability
    n_iters = 0
    error = step_tol + 1.0
    lambda_ts = zeros(N)
    # Step 1
    μ, ψ, β = c1, (1.0 - c1), log(10.0) * c2 # ψ is the branching factor, α = ψ * β
    while (error >= step_tol) && (n_iters < max_iter) # Stop iteration when optimization is smaller then tolerance 
        # Step 2
        lambda_ts[1] = 0.0
        @inbounds for i in 2:N
            lambda_ts[i] =
                exp(-β * (norm_ts[i] - norm_ts[i - 1])) * (1.0 + lambda_ts[i - 1])
        end
        lambda_ts .*= (ψ * β)
        lambda_ts .+= μ
        # Step 3
        D = 0.0 # Expected number of descendants
        div = 0.0 # Needed to calculate the new parameters for the next iteration
        @inbounds for i in 2:N
            @inbounds for j in 1:(i - 1)
                diffs = norm_ts[i] - norm_ts[j]
                D_ij = (ψ * β * exp(-β * diffs)) / lambda_ts[i] # Probability that t_i is a descendant of t_j
                D += D_ij
                div += (diffs * D_ij)
            end
        end
        # Steps 4 and 5
        error = max(abs(μ - (1 - (D / N))), abs(ψ - (D / N)), abs(β - (D / div)))
        n_iters += 1
        μ, ψ, β = 1 - (D / N), D / N, D / div # Update parameters for the next iteration
    end
    n_iters >= max_iter &&
        @warn("Maximum number of iterations reached without convergence.")
    return HawkesProcess(μ * (N / T), ψ * β * (N / T), β * (N / T)) # Unnormalize parameters
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
        A[i] = exp(-hp.β * (h.times[i] - h.times[i - 1])) * (1 + A[i - 1])
    end
    A[end] = exp(-hp.β * (h.tmax - h.times[end])) * (1 + A[end - 1]) # Used to calculate the integral of the intensity at every event time
    times = hp.μ .* (h.times .- h.tmin) # Transformation with respect to base rate
    T_base = hp.μ * duration(h) # Contribution of base rate to total length of time re-scaled process
    @inbounds for i in eachindex(times)
        times[i] += (hp.α / hp.β) * ((i - 1) - A[i]) # Add contribution of activation functions
    end
    return History(times, h.marks, 0.0, T_base + ((hp.α / hp.β) * (N - A[end]))) # A time re-scaled process starts at t=0
end

function ground_intensity(hp::HawkesProcess, h::History, t)
    activation = sum(exp.(hp.β .* (@view h.times[1:(searchsortedfirst(h.times, t) - 1)])))
    return hp.μ + (hp.α * activation / exp(hp.β * t))
end

function integrated_ground_intensity(hp::HawkesProcess, h::History, tmin, tmax)
    times = event_times(h, h.tmin, tmax)
    integral = 0.0
    for ti in times
        # Integral of activation function. 'max(tmin - ti, 0)' corrects for events that occurred
        # inside or outside the interval [tmin, tmax].
        integral += (exp(-hp.β * max(tmin - ti, 0)) - exp(-hp.β * (tmax - ti)))
    end
    integral *= hp.α / hp.β
    integral += hp.μ * (tmax - tmin) # Integral of base rate
    return integral
end

function DensityInterface.logdensityof(hp::HawkesProcess, h)
    l = 0.0
    λ_t = 0.0
    @inbounds for i in 2:length(h.times)
        λ_t = exp(-hp.β * (h.times[i] - h.times[i - 1])) * (1.0 + λ_t)
        l += λ_t
    end
    l *= hp.α
    l += hp.μ * duration(h)
    l -= integrated_ground_intensity(hp, h, h.tmin, h.tmax)
    return l
end

# Internal function for simulating Hawkes processes
function generate_descendants(rng::AbstractRNG, immigrants::Vector, T, α, β)
    descendants = eltype(immigrants)[]
    next_gen = immigrants
    a_over_b = α / β
    b_over_a = β / α
    b_inv = 1.0 / β
    while ~isempty(next_gen)
        last_gen = copy(next_gen)
        next_gen = eltype(immigrants)[]
        for parent in last_gen
            activation_integral = a_over_b * (1.0 - exp(β * (parent - T)))
            sim_transf = simulate_poisson(rng, 1.0, activation_integral)
            @. sim_transf = parent - (b_inv * log(1.0 - (b_over_a * sim_transf))) # Inverse of integral of the activation function
            append!(next_gen, sim_transf)
        end
        append!(descendants, next_gen)
    end
    return descendants
end
