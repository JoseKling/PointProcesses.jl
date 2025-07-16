struct BootstrapTest <: PPGoFTest
    n_sims::Int
    stat::Float64
    sim_stats::Vector{Float64}
end

pvalue(bs::BootstrapTest) = (count(>=(bs.stat), bs.sim_stats) + 1) / (bs.n_sims + 1)

function BootstrapTest(S::Type{<:Statistic}, pp::AbstractPointProcess, h::History; n_sims=1000)
    pp_est = estimate(pp, h)
    stat = statistic(S, pp_est, h)
    sim_stats = Vector{Float64}(undef, n_sims)
    Threads.@threads for i in 1:n_sims
        sim = simulate(pp_est, h.tmin, h.tmax)
        sim_est = fit(typeof(pp), sim)
        sim_stats[i] = statistic(S, sim_est, sim)
    end
    return BootstrapTest(n_sims, stat, sim_stats)
end