"""
    AbstractPointProcess

Common interface for all temporal point processes.
"""
abstract type AbstractPointProcess end

"""
    AbstractUnivariateProcess

Abstract type for univariate temporal point processes.
"""
abstract type AbstractUnivariateProcess <: AbstractPointProcess end

"""
    AbstractMultivariateProcess

Abstract type for multivariate temporal point processes.
"""
abstract type AbstractMultivariateProcess <: AbstractPointProcess end

@inline DensityInterface.DensityKind(::AbstractPointProcess) = HasDensity()

"""
    Base.ndims(pp)

Return the number of dimensions for a temporal point process `pp`.
"""
Base.ndims(::AbstractPointProcess)

Base.ndims(::AbstractUnivariateProcess) = 1

## Intensity functions

"""
    ground_intensity(pp, h, t)

Compute the ground intensity for a temporal point process `pp` applied to history `h` at time `t`.
For multivariate processes, it returns a vector of ground intensities for each dimension.

ground_intensity(pp, h, t, d) computes the ground intensity for a multivariate process `pp` at dimension `d`.

The ground intensity quantifies the instantaneous risk of an event with any mark occurring at time `t` after history `h`:
```
λg(t|h) = Σₘ λ(t,m|h)
```
"""
function ground_intensity end

"""
    mark_distribution(pp, t, h)

Compute the distribution of marks for a temporal point process `pp` knowing that an event takes place at time `t` after history `h`.
For multivariate processes, it returns a vector of mark distributions for each dimension.

mark_distribution(pp, h, t, d) computes the mark distribution for a multivariate process `pp` at dimension `d`.
"""
mark_distribution(pp::AbstractPointProcess, t, h) = mark_distribution(pp.mark_dist, t, h)

"""
    intensity(pp, m, t, h)

Compute the conditional intensity for a temporal point process `pp` applied to history `h` and event `(t, m)`.
For multivariate processes, it returns a vector of intensities for each dimension.

intensity(pp, h, t, d) computes the intensity for a multivariate process `pp` at dimension `d`.

The conditional intensity function `λ(t,m|h)` quantifies the instantaneous risk of an event with mark `m` occurring at time `t` after history `h`.
"""
function intensity(pp::AbstractPointProcess, m, t, h)
    return ground_intensity(pp, t, h) * densityof(pp.mark_dist, t, h, m)
end

"""
    log_intensity(pp, m, t, h)

Compute the logarithm of the conditional intensity for a temporal point process `pp` applied to history `h` and event `(t, m)`.
For multivariate processes, it returns a vector of log intensities for each dimension.

log_intensity(pp, h, t, d) computes the log intensity for a multivariate process `pp` at dimension `d`.
"""
function log_intensity(pp::AbstractPointProcess, m, t, h)
    return log(intensity(pp, m, t, h))
end

## Simulation

"""
    ground_intensity_bound(pp, t, h)

Compute a local upper bound on the ground intensity for a temporal point process `pp` applied to history `h` at time `t`.

Return a tuple of the form `(B, L)` satisfying `λg(t|h) ≤ B` for all `u ∈ [t, t+L)`.
For multivariate processes, it returns a list of tuples [(B₁, L₁), (B₂, L₂), ...] for each dimension.

ground_intensity_bound(pp, h, t, d) computes a local upper bound for a multivariate process `pp` at dimension `d`.
"""
function ground_intensity_bound end

## Learning

"""
    integrated_ground_intensity(pp, h, a, b)

Compute the integrated ground intensity (or compensator) `Λ(t|h)` for a temporal point process `pp` applied to history `h` on interval `[a, b)`:
```
Λ(h) = ∫ λg(t|h) dt
```

For multivariate processes, it returns a vector of integrated ground intensities for each dimension.

integrated_ground_intensity(pp, h, a, b, d) computes the integrated ground intensity for a multivariate process `pp` at dimension `d`.
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
function DensityInterface.logdensityof(pp::AbstractPointProcess, h::History)
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

function time_change(h::History, pp::AbstractPointProcess)
    Λ(t) = integrated_ground_intensity(pp, h, min_time(h), t)
    return time_change(h, Λ)
end
