function time_change(hp::UnivariateHawkesProcess, h::History{R,M}) where {R<:Real,M}
    n = nb_events(h)
    A = zero(R)
    sum_marks = zero(R)
    times_transformed = zeros(R, n)
    # Loop to calculate i -> ∑_{j<i} mⱼ (1 - exp(-ω (tᵢ - tⱼ)))
    for i in 2:n
        Δ = h.times[i] - h.times[i - 1]
        e = exp(-hp.ω * Δ)
        A = e * (h.marks[i - 1] + A)
        sum_marks += h.marks[i - 1]
        times_transformed[i] = sum_marks - A
    end
    A = e * (h.marks[end] + A)
    sum_marks += h.marks[end]
    times_transformed .*= hp.α / hp.ω      # Integral of activation functions
    times_transformed .+= h.times .* hp.μ  # Add integral of base rate
    tmax_transformed = (hp.α / hp.ω) * (sum_marks - A)
    tmax_transformed += R(hp.μ * duration(h))
    return History(;
        times=times_transformed,
        marks=h.marks,
        tmin=zero(R),
        tmax=tmax_transformed,
        check_args=false,
    ) # A time re-scaled process starts at t=0
end
