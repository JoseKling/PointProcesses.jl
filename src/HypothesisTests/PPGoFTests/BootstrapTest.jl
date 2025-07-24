"""
    BootstrapTest <: PPGoFTest

An object containing the results of a bootstrap-based goodness-of-fit test.

# Fields

- `n_sims::Int`: number of bootstrap simulations performed
- `stat::Float64`: observed test statistic value
- `sim_stats::Vector{Float64}`: test statistics from bootstrap simulations
"""
struct BootstrapTest <: PPGoFTest
    n_sims::Int
    stat::Float64
    sim_stats::Vector{Float64}
end

"""
    pvalue(bs::BootstrapTest) -> Float64

Calculate the p-value for a BootstrapTest.

The p-value is computed as:
    p = (#{bootstrap statistics ≥ observed statistic} + 1) / (n_sims + 1)

The +1 terms implement a conservative correction that ensures the p-value is never 0,
which would incorrectly suggest absolute certainty in rejecting the null hypothesis.

# Arguments

- `bs::BootstrapTest`: the bootstrap test result object

# Returns

- `Float64`: p-value in [0, 1], where small values provide evidence against the null hypothesis
"""
pvalue(bs::BootstrapTest) = (count(>=(bs.stat), bs.sim_stats) + 1) / (bs.n_sims + 1)

"""
    BootstrapTest(S::Type{<:Statistic}, pp::Type{<:AbstractPointProcess}, h::History; n_sims=1000) -> BootstrapTest

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

# Notes

- Uses `Threads.@threads` for parallel computation of bootstrap statistics
- More computationally intensive than `NoBootstrapTest` due to parameter re-estimation

# Example

```julia
# Bootstrap test for Hawkes process model adequacy
test = BootstrapTest(KSExponential, HawkesProcess(1.0, 1.0, 2.0), history; n_sims=1000)
p = pvalue(test)
```
"""
function BootstrapTest(S::Type{<:Statistic}, PP::Type{<:AbstractPointProcess}, h::History; n_sims=1000)
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