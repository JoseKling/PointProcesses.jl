function Base.rand(rng::AbstractRNG, pp::AbstractPoissonProcess, tmin::Real, tmax::Real)
    mark_dist = mark_distribution(pp)
    N = rand(rng, Poisson(float(ground_intensity(pp) * (tmax - tmin))))
    times = sort!(rand(rng, Uniform(tmin, tmax), N))
    marks = [rand(rng, mark_dist) for n in 1:N]
    return History(; times=times, marks=marks, tmin=tmin, tmax=tmax)
end

function Base.rand(pp::AbstractPoissonProcess, tmin::Real, tmax::Real)
    return rand(default_rng(), pp, tmin, tmax)
end
