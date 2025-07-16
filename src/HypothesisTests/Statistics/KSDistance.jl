struct KSDistance{T<:UnivariateDistribution} end
KSDistance(T::Type{<:UnivariateDistribution}) = KSDistance{T}()

function statistic(::KSDistance{Exponential}, pp::AbstractPointProcess, h::History)
    (length(h.times) < 2) && return 1.0 # If `h` has only 2 elements, than there are no interevent times
    X = diff(time_change(pp, h).times) # X → sorted time re-scaled inter event times
    sort!(X)
    return ksstats(X, Exponential)[2]
end

function statistic(::KSDistance{Uniform}, pp::AbstractPointProcess, h::History)
    (length(h.times) < 1) && return 1.0 # No events ⇒ maximum distance
    transf = time_change(pp, h).times # transf → time re-scaled event times
    return ksstats(transf.times, Uniform(transf.tmin, transf.tmax))[2]
end

