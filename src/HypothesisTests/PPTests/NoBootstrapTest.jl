"""
    NoBootstrapTest <: PPTest

An object containing the results of a non-bootstrap based goodness-of-fit test.
The p-value of the test is calculated as
    p = (count(sim_stats ≥ stat) + 1) / (n_sims + 1).

# Fields
- `n_sims::Int`: number of simulations performed
- `stat::Float64`: observed test statistic value
- `sim_stats::Vector{Float64}`: test statistics from simulated data
"""
struct NoBootstrapTest <: PPTest
    n_sims::Int
    stat::Float64
    sim_stats::Vector{Float64}
end

function StatsAPI.pvalue(nbt::NoBootstrapTest)
    return (count(>=(nbt.stat), nbt.sim_stats) + 1) / (nbt.n_sims + 1)
end

"""
    NoBootstrapTest(S::Type{<:Statistic}, pp::AbstractPointProcess, h::History; n_sims::Int=1000, rng::AbstractRNG=default_rng())

Perform a goodness-of-fit test using simulation without bootstrap resampling, comparing
the test statistic computed on the observed data against the distribution of the same
statistic computed on data simulated from the fitted model.

If λ₀(t) is the true intensity function of the process that generated the observed
history, and λ(t; θ) is a a parametrization of the intensity, then there are two forms
for the null hypothesis:

    1. H₀: λ₀(t) = λ(t; θ₀)
    2. H₀: There exists parameters θₒ such that λ₀(t) = λ(t; θ₀)

If `pp` is an instance of an `AbstractPointProcess`, the null hypothesis 1 is considered, if
a `pp` is a `Type{<:AbstractPointProcess}`, the method uses null hypothesis 2.

Notice that this test is better suited when the parameter θ₀ is known (form 1), since this
procedure does not account for parameter estimation error. For more details on this, see
[Jogesh Babu and Rao (2004)](http://www.jstor.org/stable/25053332),
[Reynaud-Bouret et. al. (2014)](https://doi.org/10.1186/2190-8567-4-3), 
[Kling and Vetter (2025)](https://doi.org/10.1111/sjos.70029).

# Arguments
- `S::Type{<:Statistic}`: the type of test statistic to use
- `pp::Union{AbstractPointProcess, Type{<:AbstractPointProcess}}`: the null hypothesis model family
- `h::History`: the observed event history
- `n_sims::Int=1000`: number of simulations to perform for the test
- `rng::AbstractRNG=default_rng()test statistics from simulated data`: Random number generator

# Returns
- `NoBootstrapTest`: test result object containing the observed statistic, simulated statistics, and test metadata

# Example
```julia
# Test null hypothesis of form 1. Known θ₀
test = NoBootstrapTest(KSDistance(Exponential), HawkesProcess(1, 1, 2), history; n_sims=1000)
p = pvalue(test)
# Test null hypothesis of form 2. Unknown θ₀
test = NoBootstrapTest(KSDistance(Exponential), HawkesProcess, history; n_sims=1000)
p = pvalue(test)
```
"""
function NoBootstrapTest(
    S::Type{<:Statistic},
    pp::AbstractPointProcess,
    h::History;
    n_sims::Int=1000,
    rng::AbstractRNG=default_rng(),
)
    isempty(h.times) && @warn "Event history is empty."

    stat = statistic(S, pp, h)

    sim_stats = Vector{typeof(stat)}(undef, n_sims)

    # one RNG per thread, seeded deterministically from the master rng
    rngs = [Xoshiro(rand(rng, UInt)) for _ in 1:Threads.nthreads()]

    Threads.@threads for i in 1:n_sims
        tid = Threads.threadid()
        local_rng = rngs[tid]

        sim = simulate(local_rng, pp, h.tmin, h.tmax)
        sim_stats[i] = statistic(S, pp, sim)
    end
    return NoBootstrapTest(n_sims, stat, sim_stats)
end

function NoBootstrapTest(
    S::Type{<:Statistic}, PP::Type{<:AbstractPointProcess}, h::History; kwargs...
)
    if isempty(h.times)
        throw(ArgumentError("Test is not valid for empty event history."))
    end

    pp_est = fit(PP, h)
    return NoBootstrapTest(S, pp_est, h; kwargs...)
end
