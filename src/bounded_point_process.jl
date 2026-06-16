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

ground_intensity(bpp::BoundedPointProcess, args...) = ground_intensity(bpp.pp, args...)
mark_distribution(bpp::BoundedPointProcess, args...) = mark_distribution(bpp.pp, args...)
intensity(bpp::BoundedPointProcess, args...) = intensity(bpp.pp, args...)
log_intensity(bpp::BoundedPointProcess, args...) = log_intensity(bpp.pp, args...)
function ground_intensity_bound(bpp::BoundedPointProcess, args...)
    return ground_intensity_bound(bpp.pp, args...)
end
function integrated_ground_intensity(bpp::BoundedPointProcess, args...)
    return integrated_ground_intensity(bpp.pp, args...)
end
