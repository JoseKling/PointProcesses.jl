"""
    History{T<:Real, M}

Linear event histories with temporal locations of type `T` and marks of type `M`.

# Fields

- `times::Vector{T}`: sorted vector of event times
- `tmin::T`: start time
- `tmax::T`: end time
- `marks::Vector{M}`: associated vector of event marks
"""
struct History{N,T<:Real,M,D}
    times::Vector{T}
    tmin::T
    tmax::T
    marks::Vector{M}
    dims::Vector{D}

    function History{N}(times::Vector{<:Real}, tmin::Real, tmax::Real, marks::Vector, dims::Vector{<:Int}; check_args::Bool=true) where {N}
        check_args && check_hist_args(N, times, tmin, tmax, marks, dims)
        T = promote_type(eltype(times), typeof(tmin), typeof(tmax))
        M = eltype(marks)
        D = Int
        new{N,T,M,D}(times, tmin, tmax, marks, dims)
    end

    function History{1}(times::Vector{<:Real}, tmin::Real, tmax::Real, marks::Vector, ::Any; check_args::Bool=true)
        dims = fill(nothing, length(times))
        check_args && check_hist_args(1, times, tmin, tmax, marks, dims)
        T = promote_type(eltype(times), typeof(tmin), typeof(tmax))
        M = eltype(marks)
        D = Nothing
        new{1,T,M,D}(times, tmin, tmax, marks, dims)
    end
end

function History(
    times::Vector{<:Vector{<:Real}},
    tmin::Real,
    tmax::Real,
    marks::Vector{<:Vector};
    check_args=true,
)
    N = length(times)
    if check_args
        if tmin >= tmax
            throw(
                DomainError(
                    (tmin, tmax),
                    "End of interval must be strictly larger than the start.",
                ),
            )
        end
        if length(times) != length(marks) || any([length(times[i]) != length(marks[i]) for i in 1:N])
            throw(ArgumentError("`times` and `marks` must have equal shapes"))
        end
    end
    times_vec = mapreduce(i -> times[i], vcat, 1:N)
    marks_vec = mapreduce(i -> marks[i], vcat, 1:N)
    if N == 1
        dims = fill(nothing, length(times[1]))
    else
        dims = mapreduce(i -> fill(i, length(times[i])), vcat, 1:N)
    end
    if !isempty(times_vec)
        perm = sortperm(times_vec)
        times_vec .= times_vec[perm]
        marks_vec .= marks_vec[perm]
        dims .= dims[perm]
        if times_vec[1] < tmin || times_vec[end] >= tmax
            @warn "Events outside of provided interval were discarded."
            il = searchsortedfirst(times_vec, tmin)
            ir = searchsortedfirst(times_vec, tmax) - 1
            times_vec = times_vec[il:ir]
            marks_vec = marks_vec[il:ir]
            dims = dims[il:ir]
        end
    end
    return History{N}(times_vec, tmin, tmax, marks_vec, dims)
end

function History(times::Vector{<:Real}, tmin::Real, tmax::Real; check_args=true)
    return History([times], tmin, tmax; check_args=check_args)
end

function History(times::Vector{<:Real}, tmin::Real, tmax::Real, marks::Vector; check_args=true)
    return History{1}(times, tmin, tmax, marks, nothing; check_args=check_args)
end

function History(times::Vector{<:Real}, tmin::Real, tmax::Real, marks::Vector, dims::Vector; check_args=true)
    return History([times], tmin, tmax, [marks], [dims]; check_args=check_args)
end

function History(times::Vector{<:Vector{<:Real}}, tmin::Real, tmax::Real; check_args=true)
    marks = [fill(nothing, length(times[i])) for i in 1:length(times)]
    return History(times, tmin, tmax, marks; check_args=check_args)
end

function History(; times, tmin, tmax, marks=fill(nothing, length(times)), dims=fill(nothing, length(times)), check_args=true)
    N = length(unique(dims))
    if N  <= 1
        return History{1}(times, tmin, tmax, marks, nothing; check_args=check_args)
    else
        return History{N}(times, tmin, tmax, marks, dims; check_args=check_args)
    end
end

const UnivariateHistory{T<:Real,M} = History{1,T,M,Nothing}

function Base.show(io::IO, h::History{N,T,M}) where {N,T,M}
    return print(
        io, "History{$N,$T,$M} with $(nb_events(h)) events on interval [$(h.tmin), $(h.tmax))"
    )
end

function Base.show(io::IO, h::UnivariateHistory{T,M}) where {T,M}
    return print(
        io, "UnivariateHistory{$T,$M} with $(nb_events(h)) events on interval [$(h.tmin), $(h.tmax))"
    )
end

"""
    event_times(h)

Return the sorted vector of event times for `h`.
"""
event_times(h::History) = h.times

"""
    event_times(h, d)

Return event times of the `d`-th marginal process of `h`.
"""
event_times(h::History{N}, d::Int) where {N} = @view h.times[h.dims .== d]
event_times(h::History{1}, ::Int) = h.times

"""
    event_times(h, a, b)

Return the sorted vector of event times in `h` inside the interval `[a, b)`.
"""
function event_times(h::History, a, b)
    imin = searchsortedfirst(h.times, a)
    imax = searchsortedfirst(h.times, b) - 1
    return @view h.times[imin: imax]
end

"""
    event_times(h, a, b, d)

Return event times of the `d`-th marginal process of `h` inside the interval `[a, b)`.
"""
function event_times(h::History, a::Real, b::Real, d::Int)
    imin = searchsortedfirst(h.times, a)
    imax = searchsortedfirst(h.times, b) - 1
    inds = [i for i in imin:imax if h.dims[i] == d]
    return @view h.times[inds]
end

"""
    min_time(h)

Return the starting time of `h` (not the same as the first event time).
"""
min_time(h::History) = h.tmin

"""
    max_time(h)

Return the end time of `h` (not the same as the last event time).
"""
max_time(h::History) = h.tmax

"""
    event_marks(h)

Return the vector of event marks for `h`, sorted according to their event times.
"""
event_marks(h::History) = h.marks

"""
    event_marks(h, tmin, tmax)

Return the sorted vector of marks of events between `a` and `b` in `h`.
"""
function event_marks(h::History, a, b)
    imin = searchsortedfirst(h.times, a)
    imax = searchsortedfirst(h.times, b) - 1
    return @view h.marks[imin: imax]
end

"""
    max_mark(h; [init])

Return the largest event mark if it is larger than `init`, and `init` otherwise.
"""
max_mark(h::History; init=first(event_marks(h))) = maximum(event_marks(h); init=init)

"""
    min_mark(h; [init])

Return the smallest event mark if it is smaller than `init`, and `init` otherwise.
"""
min_mark(h::History; init=first(event_marks(h))) = minimum(event_marks(h); init=init)

"""
    event_dims(h)

Return the vector containing in which marginal process an event of `h` occurred,
sorted according to their event times.
"""
event_dims(h::History) = h.dims

"""
    event_dims(h, tmin, tmax)

Return the sorted vector of marginal processes between `a` and `b` in `h`.
"""
function event_dims(h::History, a, b)
    imin = searchsortedfirst(h.times, a)
    imax = searchsortedfirst(h.times, b) - 1
    return @view h.dims[imin: imax]
end

"""
    n_dims(h)

Return the nuber of dimensions in the event history `h`
"""
n_dims(::History{N}) where {N} = N

Base.length(h::History) = nb_events(h)

"""
    nb_events(h)

Count events in `h`.
"""
nb_events(h::History) = length(h.times)

"""
    nb_events(h, d)

Count events in `d`-th marginal process of `h`.
"""
nb_events(h::History{1}, ::Int) = nb_events(h)
nb_events(h::History{N}, d::Int) where {N} = sum(h.dims .== d)

"""
    nb_events(h, a, b)

Count events in `h` during the interval `[a, b)`.
"""
function nb_events(h::History, a, b)
    imin = searchsortedfirst(event_times(h), a)
    imax = searchsortedfirst(event_times(h), b)
    return imax - imin
end

"""
    nb_events(h, d, a, b)

Count events in `d`-th marginal process of `h` during interval [a, b).
"""
nb_events(h::History{1}, a, b, ::Int) = nb_events(h, a ,b)
function nb_events(h::History{N}, a, b, d::Int) where {N} 
    imin = searchsortedfirst(h.times, a)
    imax = searchsortedfirst(h.times, b) - 1
    return mapreduce(i -> h.dims[i] == d, +, imin:imax)
end

"""
    has_events(h)

Check the presence of events in `h`.
"""
has_events(h::History) = nb_events(h) > 0

Base.isempty(h::History) = !has_events(h)

"""
    has_events(h, a, b)

Check the presence of events in `h` during the interval `[a, b)`.
"""
has_events(h::History, a, b) = nb_events(h, a, b) > 0

"""
    duration(h)

Compute the difference `h.tmax - h.tmin`.
"""
duration(h::History) = max_time(h) - min_time(h)

"""
    push!(h, t, m, k)

Add event `(t, m)` inside the interval `[h.tmin, h.tmax)` in marginal process `k` at the end of history `h`.
"""
function Base.push!(h::History, t::Real, m, d; check_args=true)
    if check_args
        @assert h.tmin <= t < h.tmax
        @assert (length(h) == 0) || (h.times[end] <= t)
    end
    push!(h.times, t)
    push!(h.marks, m)
    push!(h.dims, d)
    return nothing
end

function Base.push!(h::UnivariateHistory, t::Real, m; check_args=true)
    push!(h, t, m, nothing; check_args=check_args)
    return nothing
end

"""
    append!(h, ts, ms, ks)

Append events `(ts, ms)` inside the interval `[h.tmin, h.tmax)` in marginal processes `ks` at the end of history `h`.
"""
function Base.append!(h::History, ts::Vector{<:Real}, ms, ds; check_args=true)
    if check_args
        perm = sortperm(ts)
        ts .= ts[perm]
        ms .= ms[perm]
        ds .= ds[perm]
        @assert h.tmin <= ts[1] && ts[end] < h.tmax
        @assert (length(h) == 0) || (h.times[end] <= ts[1])
    end
    append!(h.times, ts)
    append!(h.marks, ms)
    append!(h.dims, ds)
    return nothing
end

function Base.append!(h::History{1}, ts::Vector{<:Real}, ms; check_args=true)
    append!(h, ts, ms, fill(nothing, length(ts)); check_args=check_args)
end
"""
    cat(h1, h2)

If h1 and h2 are consecutive event histories, i.e., the end of
h1 coincides with the beginning of h2, then create a new event
history by concatenating h1 and h2.
"""
function Base.cat(h1::History{N}, h2::History{N}) where {N}
    max_time(h1) ≈ min_time(h2) || throw(
        DomainError(
            (h1.tmax, h2.tmin),
            "End of h1's interval must coincide with start of h2's interval",
        ),
    )
    times = vcat(h1.times, h2.times)
    marks = vcat(h1.marks, h2.marks)
    dims = vcat(h1.dims, h2.dims)
    return History{N}(times, h1.tmin, h2.tmax, marks, dims; check_args=false)
end

function Base.cat(h1::History{N}, h2::History{M}) where {N,M}
    @error "Dimensionalities must coincide"
end

"""
    time_change(h, Λ)

Apply the time rescaling `t -> Λ(t)` to history `h`.
"""
function time_change(h::History, Λ)
    T = eltype(h.times)
    Λ0 = Λ(min_time(h))
    new_times = T.(Λ.(event_times(h)) .- Λ0)
    new_marks = copy(event_marks(h))
    new_tmin = zero(T)
    new_tmax = T(Λ(max_time(h)) - Λ0)
    return History{1}(new_times, new_tmin, new_tmax, new_marks, h.dims; check_args=false)
end

"""
    split_into_chunks(h, chunk_duration)

Split `h` into a vector of consecutive histories with individual duration `chunk_duration`.
"""
function split_into_chunks(h::History{N,T,M,D}, chunk_duration) where {N,T,M,D}
    chunks = History{N,T,M,D}[]
    limits = collect(min_time(h):chunk_duration:max_time(h))
    if !(limits[end] ≈ max_time(h))
        push!(limits, max_time(h))
    end
    for (a, b) in zip(limits[1:(end - 1)], limits[2:end])
        times = copy(event_times(h, a, b))
        marks = copy(event_marks(h, a, b))
        dims = copy(event_dims(h, a, b))
        chunk = History{N}(times, a, b, marks, dims)
        push!(chunks, chunk)
    end
    return chunks
end

#=
    check_hist_args(N, times, tmin, tmax, marks, dims)

Check validity of arguments for constructing a History with N dimensions.
=#
function check_hist_args(N::Int, times::Vector{<:Real}, tmin::Real, tmax::Real, marks, dims)
    if tmin >= tmax
        throw(
            DomainError(
                (tmin, tmax),
                "End of interval must be strictly larger than the start.",
            ),
        )
    end
    if length(times) != length(marks) || length(times) != length(dims)
        throw(ArgumentError("`times`, `marks`, and `dims` must have equal lengths"))
    end
    if any(diff(times) .< 0)
        throw(ArgumentError("`times` must be sorted in non-decreasing order"))
    end
    if any(t -> t < tmin || t >= tmax, times)
        throw(ArgumentError("All event times must lie inside the interval [tmin, tmax)"))
    end
    if N > 1
        if any(d -> d < 1 || d > N, dims)
            throw(ArgumentError("All event dimensions must lie between 1 and N"))
        end
    end
end