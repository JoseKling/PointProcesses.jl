function one_parent!(
    rng::AbstractRNG, h::History{T}, hp::UnmarkedUnivariateHawkesProcess, parent_t, _
) where {T<:Real}
    sim_transf = descendants(rng, parent_t, hp.α, hp.ω, h.tmax)
    append!(h.times, sim_transf)
    append!(h.marks, fill(nothing, length(sim_transf)))
    return nothing
end
