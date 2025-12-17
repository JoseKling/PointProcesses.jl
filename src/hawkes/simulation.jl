function simulate(rng::AbstractRNG, hp::HawkesProcess, tmin::Real, tmax::Real)
    h = simulate(rng, PoissonProcess(hp.μ, hp.mark_dist), tmin, tmax)
    generate_descendants!(rng, h, hp) # Recursively generates descendants from first events
    perm = sortperm(h.times)
    h.times .= h.times[perm]
    h.marks .= h.marks[perm]
    return h
end

function simulate(hp::HawkesProcess, tmin::Real, tmax::Real)
    return simulate(default_rng(), hp, tmin, tmax)
end

#=
Internal function for simulating Hawkes processes
The history `h` contains the event times generated from the base rate of the process, the
"parents" of the process. This is the starting generation, gen_0.
For each t_g ∈ gen_n, simulate an inhomogeneous Poisson process over the interval [t_g, T]
with intensity λ(t) = α exp(-ω(t - t_g)) using the inverse method.
gen_{n+1} is the set of all events simulated from all events in gen_n.
The algorithm stops when the simulation from one generation results in no further events.
=#
function generate_descendants!(
    rng::AbstractRNG, h::History{T,M}, hp::HawkesProcess
) where {T<:Real,M}
    gen_start = 1              # Starting index of generation gen_0
    gen_end = nb_events(h)     # Last index of generation gen_0
    while gen_start <= gen_end # As long as the current genertion is not empty, generate more descendants
        for parent_idx in gen_start:gen_end
            # generates the descendants of one single element of the current generation
            one_parent!(rng, h, hp, h.times[parent_idx], h.marks[parent_idx]) # dispatch on the type of `hp`
        end
        # gen_i+1 are the descendants of gen_i
        gen_start = gen_end + 1
        gen_end = nb_events(h)
    end
    return nothing
end

# generates the descendants of one single parent using the inverse transform method
function descendants(rng::AbstractRNG, parent::R, α::Real, ω::Real, tmax::R) where {R<:Real}
    T = float(R)
    αT = T(α)
    ωT = T(ω)
    activation_integral = (αT / ωT) * (one(T) - exp(ωT * (parent - tmax)))
    sim_transf = simulate_poisson_times(rng, one(T), zero(T), activation_integral)
    @. sim_transf = parent - (inv(ωT) * log(one(T) - ((ωT / αT) * sim_transf))) # Inverse of integral of the activation function
    return sim_transf
end
