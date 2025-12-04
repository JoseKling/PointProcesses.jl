function ground_intensity(hp::UnmarkedUnivariateHawkesProcess, h::History, ts)
    return hp.μ .+ (hp.α .* A_Ozaki_ts(h.times, fill(nothing, length(h.times)), hp.ω, ts))
end

function intensity(
    hp::UnmarkedUnivariateHawkesProcess{R}, _, ts, h::History
) where {R<:Real}
    return ground_intensity(hp, h, ts)
end

intensity(hp::UnmarkedUnivariateHawkesProcess, ts, h::History) = ground_intensity(hp, h, ts)

function integrated_ground_intensity(hp::UnmarkedUnivariateHawkesProcess, h::History, ts)
    A = A_Ozaki_ts(h.times, fill(nothing, length(h.times)), hp.ω, ts)
    ns = map(t -> nb_events(h, h.tmin, t), ts)
    return (hp.μ .* ts) .+ sum((hp.α / hp.ω) .* (ns .- A))
end

function integrated_ground_intensity(hp::UnmarkedUnivariateHawkesProcess, h::History, a, b)
    return integrated_ground_intensity(hp, h, b) - integrated_ground_intensity(hp, h, a)
end

# logdensityof (log-likelihood)
function DensityInterface.logdensityof(hp::UnmarkedUnivariateHawkesProcess, h::History)
    A = 0.0
    sum_logλ = 0.0
    for i in 2:nb_events(h)
        A = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + A)
        sum_logλ += log(hp.μ + hp.α * A)
    end
    A = exp(-hp.ω * (h.tmax - h.times[end])) * (1 + A)
    Λ_T = (hp.μ * duration(h)) + (hp.α / hp.ω) * (nb_events(h) - A)
    return sum_logλ - sum(Λ_T)
end
