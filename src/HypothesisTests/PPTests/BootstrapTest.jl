"""
    BootstrapTest <: PPTest

An object containing the results of a bootstrap-based goodness-of-fit test.
The p-value of the test is calculated as
    p = (count(sim_stats ≥ stat) + 1) / (n_sims + 1).

# Fields
- `n_sims::Int`: number of bootstrap simulations performed
- `stat::Float64`: observed test statistic value
- `sim_stats::Vector{Float64}`: test statistics from bootstrap simulations
"""
struct BootstrapTest <: PPTest
    n_sims::Int
    stat::Float64
    sim_stats::Vector{Float64}
end

function StatsAPI.pvalue(bs::BootstrapTest)
    (count(>=(bs.stat), bs.sim_stats) + 1) / (bs.n_sims + 1)
end

"""
    BootstrapTest(S::Type{<:Statistic}, pp::Type{<:AbstractPointProcess}, h::History; n_sims=1000)

Perform a goodness-of-fit test using simulation with bootstrap resampling, comparing
the test statistic computed on the observed data against the distribution of the same
statistic computed on data simulated from the fitted model.

If λ₀(t) is the true intensity function of the process that generated the observed
history, and λ(t; θ) is a a parametrization of the intensity, then the null hypothesis is

    H₀: There exists parameters θₒ such that λ₀(t) = λ(t; θ₀)

This procedure is specifically aimed for testing hypotheses where parameters need to
be estimated. Details are provided in [Kling and Vetter (2024)](https://arxiv.org/abs/2407.09130).

# Arguments
- `S::Type{<:Statistic}`: the type of test statistic to use
- `pp::Type{<:AbstractPointProcess}`: the null hypothesis model family
- `h::History`: the observed event history
- `n_sims::Int=1000`: number of bootstrap simulations to perform

# Returns
- `BootstrapTest`: test result object containing the observed statistic, bootstrap statistics, and test metadata

# Example

```julia
# Bootstrap test for Hawkes process model adequacy
test = BootstrapTest(KSDistance(Exponential), HawkesProcess, history; n_sims=1000)
p = pvalue(test)
```
"""
function BootstrapTest(
    S::Type{<:Statistic}, PP::Type{<:AbstractPointProcess}, h::History; n_sims=1000
)
    pp_est = estimate(PP, h)
    stat = statistic(S, pp_est, h)
    sim_stats = Vector{Float64}(undef, n_sims)
    Threads.@threads for i in 1:n_sims
        sim = simulate(pp_est, h.tmin, h.tmax)
        sim_est = fit(PP, sim)
        sim_stats[i] = statistic(S, sim_est, sim)
    end
    return BootstrapTest(n_sims, stat, sim_stats)
end

function BootstrapTest(
    S::Type{<:Statistic}, pp::AbstractPointProcess, h::History; kwargs...
)
    BootstrapTest(default_rng(), S, pp, h; kwargs...)
end