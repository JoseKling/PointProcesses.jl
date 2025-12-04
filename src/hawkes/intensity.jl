#=
Calculates the vector `A` as is [Ozaki (1979)](https://doi.org/10.1007/bf02480272)
A[1] = 0
A[i] = Σ_{tᵢ<tⱼ} exp(-ω(tᵢ - tⱼ))

For marked processes, the calculation is slightly different
A[1] = 0
A[i] = Σ_{tᵢ<tⱼ} mᵢ exp(-ω(tᵢ - tⱼ)), where mᵢ is the i-th mark of the process

This is used to efficiently calculate

    λ(tᵢ) = μ + α ∑_{j<i} mⱼ exp(-ω(tᵢ - tⱼ)) = μ + α A[i]
=#
function A_Ozaki(times, marks, ω)
    A = zeros(length(times))
    for i in 2:length(times)
        A[i] = update_A(A[i - 1], exp(-ω * (times[i] - times[i - 1])), marks[i - 1])
    end
    return A
end

#=
Uses the idea from [Ozaki (1979)](https://doi.org/10.1007/bf02480272)
to efficiently calculate the value of the intensity function at any
time. If tₙ is the last event before time s, then

    λ(s) = μ + α exp(-ω (s - tₙ)) (mₙ + A[n])

The vector A is only calculated up to n.
=#
function A_Ozaki_ts(
    times::AbstractVector{R1}, marks::AbstractVector{M}, ω::R2, t::R3
) where {R1<:Real,R2<:Real,R3<:Real,M}
    T = M == Nothing ? float(promote_type(R1, R2, R3)) : float(promote_type(R1, R2, R3, M))
    A = zero(T)
    (isempty(times) || t <= times[1]) && return A
    ind_times = 2
    while ind_times <= length(times) && times[ind_times] < t
        A = update_A(
            A, exp(-ω * (times[ind_times] - times[ind_times - 1])), marks[ind_times - 1]
        )
        ind_times += 1
    end
    A = update_A(A, exp(-ω * (t - times[ind_times - 1])), marks[ind_times - 1])
    return A
end

#=
Uses the idea from [Ozaki (1979)](https://doi.org/10.1007/bf02480272)
to efficiently calculate the values of the intensity function at any
**ORDERED** set of times.
=#
function A_Ozaki_ts(
    times::AbstractVector{R1}, marks::AbstractVector{M}, ω::R2, ts::AbstractVector{<:R3}
) where {R1<:Real,R2<:Real,R3<:Real,M}
    if M == Nothing
        T = float(promote_type(R1, R2, R3))
    else
        T = float(promote_type(R1, R2, R3, M))
    end
    A = zeros(T, length(ts))
    (isempty(times) || ts[end] <= times[1] || isempty(ts)) && return A
    ind_A = 1
    while ts[ind_A] <= times[1] # A[i] = 0 for all ts before the first event
        ind_A += 1
    end
    ind_times = 2
    while ind_A < length(ts)
        while ind_times <= length(times) && times[ind_times] < ts[ind_A]
            A[ind_A] = update_A(
                A[ind_A],
                exp(-ω * (times[ind_times] - times[ind_times - 1])),
                marks[ind_times - 1],
            )
            ind_times += 1
        end
        A[ind_A + 1] = A[ind_A]
        A[ind_A] = update_A(
            A[ind_A], exp(-ω * (ts[ind_A] - times[ind_times - 1])), marks[ind_times - 1]
        )
        ind_A += 1
    end
    if ind_times > length(times) || times[ind_times] >= ts[end]
        A[end] = update_A(
            A[end], exp(-ω * (ts[end] - times[ind_times - 1])), marks[ind_times - 1]
        )
    else
        A[end] = update_A(
            A[end],
            exp(-ω * (times[ind_times] - times[ind_times - 1])),
            marks[ind_times - 1],
        )
        A[end] = update_A(A[end], exp(-ω * (ts[end] - times[ind_times])), marks[ind_times])
    end
    return A
end

function update_A(Ai_1, eωt::Real, _::Nothing)
    return eωt * (1 + Ai_1)
end

function update_A(Ai_1, eωt::Real, mi_1::Real)
    return eωt * (mi_1 + Ai_1)
end
