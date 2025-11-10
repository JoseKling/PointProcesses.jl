"""
    PPTest

Interface for all goodness-of-fit tests
"""
abstract type PPTest <: HypothesisTest end

"""
    pvalue(test::PPTest)

Calculate the p-value of a goodness-of-fit test on a process.

# Arguments
- `bs::BootstrapTest`: the bootstrap test result object

# Returns
- `Float64`: p-value in [0, 1], where small values provide evidence against the null hypothesis
"""
function StatsAPI.pvalue(::PPTest) end

"""
    Statistic

Interface for all test statistics.
"""
abstract type Statistic end

"""
    statistic(::Statistic, pp::AbstractPointProcess, h::History)

Compute the value of the test statistic with respect to `pp` and `h`.

# Arguments
- `::Statistic`: The type of test statistic to be computed
- `pp::AbstractPointProcess`: null-hypothesis model for the event history `h`
- `h::History`: the observed event history

# Returns
- `Float64`: the resulting test statistic

# Example

```julia
#=
Calculate the Kolmogorov-Smirnov distance between the distribution of the
time-changed event times of `h` and a standard exponential.
=#
ks_stat = statistic(KSDistance(Exponential), hawkes_process, history)
```
"""
function statistic end
