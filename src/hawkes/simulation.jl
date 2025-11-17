function simulate(rng::AbstractRNG, hp::HawkesProcess, tmin, tmax)
    sim = simulate_poisson_times(rng, hp.μ, tmin, tmax) # Simulate Poisson process with base rate
    sim_desc = generate_descendants(rng, sim, tmax, hp.α, hp.ω) # Recursively generates descendants from first events
    append!(sim, sim_desc)
    sort!(sim)
    return History(; times=sim, tmin=tmin, tmax=tmax, check=false)
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
