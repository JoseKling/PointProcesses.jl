function time_change(hp::UnivariateHawkesProcess{<:Real,D}, h::History{R}) where {R<:Real,D}
    T = float(R)
    n = nb_events(h)
    A = zero(T)
    μ, α, ω = T.((hp.μ, hp.α, hp.ω))
    sum_marks = zero(T)
    times_transformed = zeros(T, n)

    # Loop to calculate i -> ∑_{j<i} mⱼ (1 - exp(-ω (tᵢ - tⱼ)))
    for i in 2:n
        mi_1 = T(process_mark(D, h.marks[i - 1]))
        A = update_A(mi_1 * α, ω, h.times[i], h.times[i - 1], A)
        sum_marks += mi_1
        times_transformed[i] = sum_marks - A
    end

    # Calculate ∑_{i=1}^n mᵢ (1 - exp(-ω (tmax - tᵢ)))
    mn = T(process_mark(D, h.marks[end]))
    A = update_A(mn * α, ω, h.tmax, h.times[end], A)
    sum_marks += mn

    # Integral of activation functions
    times_transformed .*= α / ω      # Integral of activation functions
    tmax_transformed = (α / ω) * (sum_marks - A)

    # Add integral of base rate
    times_transformed .+= h.times .* μ
    tmax_transformed += μ * duration(h)

    # A time re-scaled process starts at t=0
    return History(;
        times=times_transformed,
        marks=h.marks,
        tmin=zero(T),
        tmax=tmax_transformed,
        check_args=false,
    )
end
