function simulate(rng::AbstractRNG, pp::PoissonProcess, tmin::T, tmax::T) where {T<:Real}
    times = simulate_poisson_times(rng, pp.λ, tmin, tmax)
    h_temp = History(times, tmin, tmax)
    marks = [sample_mark(pp.mark_dist, t, h_temp) for t in times]
    return History(; times=times, marks=marks, tmin=tmin, tmax=tmax)
end

# function simulate(rng::AbstractRNG, pp::PoissonProcess, tmin::T, tmax::T) where {T<:Real}
#     h = History(T[], tmin, tmax, eltype(pp.mark_dist)[])
#     inter_dist = Exponential(inv(pp.λ))
#     t = rand(rng, inter_dist)
#     while t < tmax
#         m = sample_mark(pp.mark_dist, t, h)
#         push!(h, t, m; check_args=False)
#         t = rand(rng, inter_dist)
#     end
#     return h
# end
