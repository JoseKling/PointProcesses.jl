function time_change(hp::UnmarkedUnivariateHawkesProcess, h::History{T,M}) where {T<:Real,M}
    n = nb_events(h)
    A = zeros(T, n + 1) # Array A in Ozaki (1979)
    @inbounds for i in 2:n
        A[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + A[i - 1])
    end
    A[end] = exp(-hp.ω * (h.tmax - h.times[end])) * (1 + A[end - 1]) # Used to calculate the integral of the intensity at every event time
    times = T.(hp.μ .* (h.times .- h.tmin)) # Transformation with respect to base rate
    T_base = hp.μ * duration(h) # Contribution of base rate to total length of time re-scaled process
    for i in eachindex(times)
        times[i] += (hp.α / hp.ω) * ((i - 1) - A[i]) # Add contribution of activation functions
    end
    return History(;
        times=times,
        marks=h.marks,
        tmin=zero(T),
        tmax=T(T_base + ((hp.α / hp.ω) * (n - A[end]))),
        check_args=false,
    ) # A time re-scaled process starts at t=0
end
