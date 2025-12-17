function one_parent!(
    rng::AbstractRNG, h::History, hp::UnivariateHawkesProcess, parent_t, parent_m
)
    sim_transf = descendants(rng, parent_t, (parent_m * hp.α), hp.ω, h.tmax)
    append!(h.times, sim_transf)
    append!(h.marks, [rand(rng, mark_distribution(hp)) for _ in 1:length(sim_transf)])
    return nothing
end

function one_parent!(
    rng::AbstractRNG, h::History, hp::UnmarkedUnivariateHawkesProcess, parent_t, _
)
    sim_transf = descendants(rng, parent_t, hp.α, hp.ω, h.tmax)
    append!(h.times, sim_transf)
    append!(h.marks, fill(nothing, length(sim_transf)))
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
