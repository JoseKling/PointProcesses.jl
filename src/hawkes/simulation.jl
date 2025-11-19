function simulate(rng::AbstractRNG, hp::HawkesProcess, tmin::Real, tmax::Real)
    h = simulate(rng, PoissonProcess(hp.μ, hp.mark_dist), tmin, tmax)
    sim_desc = generate_descendants!(rng, h, hp, tmax) # Recursively generates descendants from first events
    perm = sortperm(h.times)
    h.times .= h.times[perm]
    h.marks .= h.marks[perm]
    return h
end

function simulate(hp::HawkesProcess, tmin::Real, tmax::Real)
    simulate(default_rng(), hp, tmin, tmax)
end

#=
Internal function for simulating Hawkes processes
The first generation, gen_0, is the `immigrants`, which is a set of event times.
For each t_g ∈ gen_n, simulate an inhomogeneous Poisson process over the interval [t_g, T]
with intensity λ(t) = α exp(-ω(t - t_g)) with the inverse method.
gen_{n+1} is the set of all events simulated from all events in gen_n.
The algorithm stops when the simulation from one generation results in no further events.
=#
function generate_descendants!(
    rng::AbstractRNG, h::History{T,M}, hp::HawkesProcess, tmax
) where {T<:Real,M}
    gen_start = 1
    gen_end = nb_events(h)
    while gen_start <= gen_end # Last generation is not empty
        for parent_idx in gen_start:gen_end
            one_parent!(rng, h, hp, h[parent_idx])
        end
        gen_start = gen_end + 1
        gen_end = nb_events(h)
    end
    return nothing
end

function one_parent!(
    rng::AbstractRNG, h::History{T}, hp::UnivariateHawkesProcess, parent
) where {T<:Real}
    α = isnothing(parent[2]) ? hp.α : parent[2] * hp.α
    activation_integral = (α / hp.ω) * (one(T) - exp(hp.ω * (parent[1] - h.tmax)))
    sim_transf = simulate_poisson_times(rng, one(T), zero(T), activation_integral)
    @. sim_transf = parent[1] - (inv(hp.ω) * log(one(T) - ((hp.ω / α) * sim_transf))) # Inverse of integral of the activation function
    append!(h.times, sim_transf)
    append!(h.marks, [rand(rng, hp.mark_dist) for _ in 1:length(sim_transf)])
    return nothing
end

function one_parent!(
    rng::AbstractRNG, h::History{T}, mh::MultivariateHawkesProcess, parent
) where {T<:Real}
    for m in 1:length(mh)
        α = mh.α[m, parent[2]]
        ω = mh.ω[m, parent[2]]
        activation_integral = (α / ω) * (one(T) - exp(ω * (parent[1] - h.tmax)))
        sim_transf = simulate_poisson_times(rng, one(T), zero(T), activation_integral)
        @. sim_transf = parent[1] - (inv(ω) * log(one(T) - ((ω / α) * sim_transf))) # Inverse of integral of the activation function
        append!(h.times, sim_transf)
        append!(h.marks, fill(m, length(sim_transf)))
    end
    return nothing
end
