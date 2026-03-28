"""
    BoundedPointProcess{P,T} <: AbstractPointProcess{}

Temporal point process `P` with pre-defined start and end times.

Implements some fallbacks for the `AbstractPointProcess` interface which accept fewer arguments.

# Fields

- `pp::P`: underlying point process
- `tmin::T`: start time
- `tmax::T`: end time
"""
struct BoundedPointProcess{P<:AbstractPointProcess,T<:Real} <: AbstractPointProcess
    pp::P
    tmin::T
    tmax::T
end

min_time(bpp::BoundedPointProcess) = bpp.tmin
max_time(bpp::BoundedPointProcess) = bpp.tmax

"""
    simulate([rng,] bpp::BoundedPointProcess)

Simulate a point process on a predefined time interval.
"""
function simulate(rng::AbstractRNG, bpp::BoundedPointProcess)
    return simulate(rng, bpp.pp, min_time(bpp), max_time(bpp))
end

ground_intensity(bpp::BoundedPointProcess, t, h) = ground_intensity(bpp.pp, t, h)
mark_distribution(bpp::BoundedPointProcess, t, h) = mark_distribution(bpp.pp, t, h)
mark_distribution(bpp::BoundedPointProcess, t) = mark_distribution(bpp.pp, t) # For simulate_ogata
intensity(bpp::BoundedPointProcess, m, t, h) = intensity(bpp.pp, m, t, h)
log_intensity(bpp::BoundedPointProcess, m, t, h) = log_intensity(bpp.pp, m, t, h)

function ground_intensity_bound(bpp::BoundedPointProcess, t, h)
    return ground_intensity_bound(bpp.pp, t, h)
end

function integrated_ground_intensity(bpp::BoundedPointProcess, h, a, b)
    return integrated_ground_intensity(bpp.pp, h, a, b)
end
