struct NoBootstrapTest <: PPGoFTest
    n_sims::Int
    stat::Float64
    sim_stats::Vector{Float64}
end

pvalue(nbs::NoBootstrapTest) = (count(>=(nbs.stat), nbs.sim_stats) + 1) / (nbs.n_sims + 1)

function NoBootstrapTest(S::Type{<:Statistic}, pp::AbstractPointProcess, h::History; n_sims=1000)
    pp_est = estimate(pp, h)
    stat = statistic(S, pp_est, h)
    sim_stats = Vector{Float64}(undef, n_sims)
    Threads.@threads for i in 1:n_sims
        sim = simulate(pp_est, h.tmin, h.tmax)
        sim_stats[i] = statistic(S, pp_est, sim)
    end
    return NoBootstrapTest(n_sims, stat, sim_stats)
end
