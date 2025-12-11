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
#=
`stat` and `sim_stats` set to type `Float64`, because the `ksstats`
function from `HypothesisTests.jl` always returns a `Float64` value
If implementation changes, could define
`BootstrapTest{R} <: PPTest where {R<:Real}`
=#

function StatsAPI.pvalue(bt::BootstrapTest)
    (count(>=(bt.stat), bt.sim_stats) + 1) / (bt.n_sims + 1)
end

"""
    BootstrapTest(S::Type{<:Statistic}, pp::AbstractPointProcess, h::History; n_sims::Int=1000, rng::AbstractRNG=default_rng())

Perform a goodness-of-fit test using simulation with bootstrap resampling, comparing
the test statistic computed on the observed data against the distribution of the same
statistic computed on data simulated from the fitted model.

If λ₀(t) is the true intensity function of the process that generated the observed
history, and λ(t; θ) is a a parametrization of the intensity, then the null hypothesis is

    H₀: There exists parameters θₒ such that λ₀(t) = λ(t; θ₀)

This procedure is specifically aimed for testing hypotheses where parameters need to
be estimated. Details are provided in [Kling and Vetter (2025)](https://doi.org/10.1111/sjos.70029).

# Arguments
- `S::Type{<:Statistic}`: the type of test statistic to use
- `pp::Type{<:AbstractPointProcess}`: the null hypothesis model family
- `h::History`: the observed event history
- `n_sims::Int=1000`: number of bootstrap simulations to perform
- `rng::AbstractRNG=default_rng()`: Random number generator

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
    S::Type{<:Statistic},
    PP::Type{<:AbstractPointProcess},
    h::History;
    n_sims::Int=1000,
    rng::AbstractRNG=default_rng(),
)
    if isempty(h.times)
        throw(ArgumentError("Test is not valid for empty event history."))
    end

    pp_est = fit(PP, h; rng=rng)
    stat = statistic(S, pp_est, h)

    sim_stats = Vector{Float64}(undef, n_sims)

    # one RNG per thread, seeded deterministically from the master rng
    rngs = [Xoshiro(rand(rng, UInt)) for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:n_sims
        tid = Threads.threadid()
        local_rng = rngs[tid]

        sim = simulate(local_rng, pp_est, h.tmin, h.tmax)
        sim_est = fit(PP, sim, rng=local_rng)
        sim_stats[i] = statistic(S, sim_est, sim)
    end
    return BootstrapTest(n_sims, stat, sim_stats)
end
