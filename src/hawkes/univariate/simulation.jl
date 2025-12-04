function one_parent!(
    rng::AbstractRNG, h::History{T}, hp::UnivariateHawkesProcess, parent_t, parent_m
) where {T<:Real}
    sim_transf = descendants(rng, parent_t, parent_m * hp.α, hp.ω, h.tmax)
    append!(h.times, sim_transf)
    append!(h.marks, [rand(rng, mark_distribution(hp)) for _ in 1:length(sim_transf)])
    return nothing
end
