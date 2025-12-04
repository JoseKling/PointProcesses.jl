function time_change(hp::UnmarkedUnivariateHawkesProcess, h::History{R,M}) where {R<:Real,M}
    n = nb_events(h)
    A = zero(R)
    times_transformed = zeros(R, n)
    # Loop to calculate i -> ∑_{j<i} 1 - exp(-ω (tᵢ - tⱼ))
    for i in 2:n
        A = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (one(R) + A)
        times_transformed[i] = (i - 1) - A
    end
    A = exp(-hp.ω * (h.tmax - h.times[end])) * (one(R) + A)
    times_transformed .*= hp.α / hp.ω      # Integral of activation functions
    times_transformed .+= h.times .* hp.μ  # Add integral of base rate
    tmax_transformed = R((hp.α / hp.ω) * (n - A))
    tmax_transformed += R(hp.μ * duration(h))
    return History(;
        times=times_transformed,
        marks=h.marks,
        tmin=zero(R),
        tmax=tmax_transformed,
        check_args=false,
    ) # A time re-scaled process starts at t=0
end
