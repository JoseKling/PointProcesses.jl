"""
    BootstrapTest <: PointProcessTest

An object containing the results of a bootstrap-based goodness-of-fit test.
The p-value of the test is calculated as
    p = (count(sim_stats ≥ stat) + 1) / (n_sims + 1).

# Fields
- `n_sims::Int`: number of bootstrap simulations performed
- `stat::Float64`: observed test statistic value
- `sim_stats::Vector{Float64}`: test statistics from bootstrap simulations
"""
struct BootstrapTest <: PointProcessTest
    n_sims::Int
    stat::Float64
    sim_stats::Vector{Float64}
end
#=
`stat` and `sim_stats` set to type `Float64`, because the `ksstats`
function from `HypothesisTests.jl` always returns a `Float64` value
If implementation changes, could define
`BootstrapTest{R} <: PointProcessTest where {R<:Real}`
=#

function StatsAPI.pvalue(bt::BootstrapTest)
    return (count(>=(bt.stat), bt.sim_stats) + 1) / (bt.n_sims + 1)
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
be estimated. Details are provided in [Kling2025](@cite).

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

    # Estimate process and calculate statistic from data
    pp_est = fit(PP, h; rng=rng)
    stat = statistic(S, pp_est, h)

    # Initialize vector with test statistics from simulations
    sim_stats = Vector{Float64}(undef, n_sims)

    # Split sim_stats in one chunk per thread
    chunk_size = max(1, length(sim_stats) ÷ Threads.nthreads())
    chunks = Iterators.partition(sim_stats, chunk_size)

    # Lock for accessing the master rng safely
    l = ReentrantLock()

    tasks = map(chunks) do chunk
        Threads.@spawn begin

            # Local rng seeded deterministically from the master rng 
            local_rng = lock(() -> Xoshiro(rand(rng, UInt)), l)

            for i in eachindex(chunk)
                # Simulates a process and uses the process estimated from
                # the simulation to calculate the test statistic
                sim = simulate(local_rng, pp_est, h.tmin, h.tmax)
                sim_est = fit(PP, sim, rng=local_rng)
                chunk[i] = statistic(S, sim_est, sim)
            end
        end
    end
    fetch.(tasks)
    return BootstrapTest(n_sims, stat, sim_stats)
end
