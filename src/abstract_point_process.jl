"""
    AbstractPointProcess

Common interface for all temporal point processes.
"""
abstract type AbstractPointProcess end

@inline DensityInterface.DensityKind(::AbstractPointProcess) = HasDensity()

## Intensity functions

"""
    ground_intensity(pp, h, t)

Compute the ground intensity for a temporal point process `pp` applied to history `h` at time `t`.

The ground intensity quantifies the instantaneous risk of an event with any mark occurring at time `t` after history `h`:
```
λg(t|h) = Σₘ λ(t,m|h)
```
"""
function ground_intensity end

"""
    mark_distribution(pp, t, h)

Compute the distribution of marks for a temporal point process `pp` knowing that an event takes place at time `t` after history `h`.
"""
function mark_distribution end

"""
    intensity(pp, m, t, h)

Compute the conditional intensity for a temporal point process `pp` applied to history `h` and event `(t, m)`.

The conditional intensity function `λ(t,m|h)` quantifies the instantaneous risk of an event with mark `m` occurring at time `t` after history `h`.
"""
function intensity end

"""
    log_intensity(pp, m, t, h)

Compute the logarithm of the conditional intensity for a temporal point process `pp` applied to history `h` and event `(t, m)`.
"""
function log_intensity(pp::AbstractPointProcess, m, t, h)
    return log(ground_intensity(pp, t, h)) + logdensityof(mark_distribution(pp, t, h), m)
end

## Simulation

"""
    ground_intensity_bound(pp, t, h)

Compute a local upper bound on the ground intensity for a temporal point process `pp` applied to history `h` at time `t`.

Return a tuple of the form `(B, L)` satisfying `λg(t|h) ≤ B` for all `u ∈ [t, t+L)`.
"""
function ground_intensity_bound end

## Learning

"""
    integrated_ground_intensity(pp, h, a, b)

Compute the integrated ground intensity (or compensator) `Λ(t|h)` for a temporal point process `pp` applied to history `h` on interval `[a, b)`:
```
Λ(h) = ∫ λg(t|h) dt
```
"""
function integrated_ground_intensity end

"""
    logdensityof(pp, h)

Compute the log probability density function for a temporal point process `pp` applied to history `h`:
```
ℓ(h) = Σₖ log λ(tₖ|hₖ) - Λ(h)
```
The default method uses a loop over events combined with `integrated_ground_intensity`, but it should be reimplemented for specific processes if faster computation is possible.
"""
function DensityInterface.logdensityof(pp::AbstractPointProcess, h)
    l = -integrated_ground_intensity(pp, h, min_time(h), max_time(h))
    for (t, m) in zip(event_times(h), event_marks(h))
        l += log_intensity(pp, m, t, h)
    end
    return l
end

"""
    fit(::Type{PP}, h)
    fit(::Type{PP}, histories)

Fit a point process of type `PP` to one or several histories.

Not implemented by default.
"""
StatsAPI.fit

"""
    fit_map(::Type{PP}, h, prior)
    fit_map(::Type{PP}, histories, prior)

Fit a point process of type `PP` to one or several histories using maximum a posteriori with a `prior`.

Not implemented by default.
"""
function fit_map end
