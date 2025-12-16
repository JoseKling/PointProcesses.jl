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
ks_stat = statistic(KSDistance{Exponential}, hawkes_process, history)
```
"""
function statistic end
