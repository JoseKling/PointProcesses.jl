function simulate(rng::AbstractRNG, pp::PoissonProcess, tmin::T, tmax::T) where {T<:Real}
    times = simulate_poisson_times(rng, ground_intensity(pp), tmin, tmax)
    marks = [rand(rng, mark_distribution(pp)) for _ in 1:length(times)]
    return History(; times=times, marks=marks, tmin=tmin, tmax=tmax, check_args=false)
end
