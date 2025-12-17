function one_parent!(
    rng::AbstractRNG, h::History{T}, mh::MultivariateHawkesProcess, parent_t, parent_m
) where {T<:Real}
    for m in support(mh.mark_dist)
        sim_transf = descendants(rng, parent_t, mh.α[parent_m, m], mh.ω[m], h.tmax)
        append!(h.times, sim_transf)
        append!(h.marks, fill(m, length(sim_transf)))
    end
    return nothing
end
