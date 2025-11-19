# Ground intensity
function ground_intensity(hp::UnmarkedUnivariateHawkesProcess, h::History, t)
    times = event_times(h, h.tmin, t)
    activation = sum(hp.α .* exp.(hp.ω .* times))
    return hp.μ + (activation / exp(hp.ω * t))
end

function integrated_ground_intensity(
    hp::UnmarkedUnivariateHawkesProcess, h::History, tmin, tmax
)
    times = event_times(h, h.tmin, tmax)
    integral = 0
    for ti in times
        # Integral of activation function. 'max(tmin - ti, 0)' corrects for events that occurred
        # inside or outside the interval [tmin, tmax].
        integral += (exp(-hp.ω * max(tmin - ti, 0)) - exp(-hp.ω * (tmax - ti)))
    end
    integral *= hp.α / hp.ω
    integral += hp.μ * (tmax - tmin) # Integral of base rate
    return integral
end

function DensityInterface.logdensityof(hp::UnmarkedUnivariateHawkesProcess, h::History)
    A = zeros(nb_events(h)) # Vector A in Ozaki (1979)
    for i in 2:nb_events(h)
        A[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + A[i - 1])
    end
    return sum(log.(hp.μ .+ (hp.α .* A))) - # Value of intensity at each event
           (hp.μ * duration(h)) - # Integral of base rate
           ((hp.α / hp.ω) * sum(1 .- exp.(-hp.ω .* (duration(h) .- h.times)))) # Integral of each kernel
end

function ground_intensity(hp::UnivariateHawkesProcess, h::History, t)
    idx = searchsortedfirst(h.times, t) - 1
    times = (@view h.times[1:idx])
    marks = (@view h.marks[1:idx])
    activation = sum(hp.α .* marks .* exp.(hp.ω .* times))
    return hp.μ + (activation / exp(hp.ω * t))
end

function integrated_ground_intensity(hp::UnivariateHawkesProcess, h::History, tmin, tmax)
    times = event_times(h, h.tmin, tmax)
    marks = event_marks(h, h.tmin, tmax)
    integral = 0
    for (ti, mi) in zip(times, marks)
        # Integral of activation function. 'max(tmin - ti, 0)' corrects for events that occurred
        # inside or outside the interval [tmin, tmax].
        integral += mi * (exp(-hp.ω * max(tmin - ti, 0)) - exp(-hp.ω * (tmax - ti)))
    end
    integral *= hp.α / hp.ω
    integral += hp.μ * (tmax - tmin) # Integral of base rate
    return integral
end

function DensityInterface.logdensityof(hp::UnivariateHawkesProcess, h::History)
    A = zeros(nb_events(h)) # Vector A in Ozaki (1979)
    for i in 2:nb_events(h)
        A[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (h.marks[i - 1] + A[i - 1])
    end
    return sum(log.(hp.μ .+ (hp.α .* A))) - # Value of intensity at each event
           (hp.μ * duration(h)) - # Integral of base rate
           ((hp.α / hp.ω) * sum(h.marks .* (1 .- exp.(-hp.ω .* (duration(h) .- h.times))))) # Integral of each kernel
end
