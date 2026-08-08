function simulate(rng::AbstractRNG, pp::PoissonProcess, tmin::T, tmax::T) where {T<:Real}
    h = History(tmin, tmax, eltype(pp.mark_dist))
    inter_dist = Exponential(inv(pp.λ))
    t = tmin + T(rand(rng, inter_dist))
    while t < tmax
        m = sample_mark(rng, pp, t, h)
        push!(h, t, m; check_args=false)
        t += rand(rng, inter_dist)
    end
    return h
end
