"""
    History{T<:Real, M}

Linear event histories with temporal locations of type `T` and marks of type `M`.

# Fields

- `times::Vector{T}`: sorted vector of event times
- `tmin::T`: start time
- `tmax::T`: end time
- `marks::Vector{M}`: associated vector of event marks
"""
struct History{T<:Real,M}
    times::Vector{T}
    tmin::T
    tmax::T
    marks::Vector{M}

    function History(times, tmin, tmax, marks=fill(nothing, length(times)); check=true)
        if check
            tmin >= tmax && throw(
                DomainError(
                    (tmin, tmax),
                    "End of interval must be strictly larger than the start.",
                ),
            )
            length(marks) != length(times) && throw(
                DimensionMismatch("There must be the same number of events and marks.")
            )
            if !isempty(times)
                perm = sortperm(times)
                times .= times[perm]
                marks .= marks[perm]
                if times[1] < tmin || times[end] >= tmax
                    @warn "Events outside of provided interval were discarded."
                    il = searchsortedfirst(times, tmin)
                    ir = searchsortedfirst(times, tmax) - 1
                    times = times[il:ir]
                    marks = marks[il:ir]
                end
            end
        end
        T = promote_type(eltype(times), typeof(tmin), typeof(tmax))
        return new{T,eltype(marks)}(times, tmin, tmax, marks)
    end
end

function History(; times, tmin, tmax, marks=fill(nothing, length(times)), check=true)
    History(times, tmin, tmax, marks; check)
end

function Base.show(io::IO, h::History{T,M}) where {T,M}
    return print(
        io, "History{$T,$M} with $(nb_events(h)) events on interval [$(h.tmin), $(h.tmax))"
    )
end

"""
    event_times(h)

Return the sorted vector of event times for `h`.
"""
event_times(h::History) = h.times

"""
    event_times(h, tmin, tmax)

Return the sorted vector of event times between `tmin` and `tmax` in `h`.
"""
function event_times(h::History, tmin, tmax)
    @view h.times[searchsortedfirst(h.times, tmin):(searchsortedfirst(h.times, tmax) - 1)]
end

"""
    event_marks(h)

Return the vector of event marks for `h`, sorted according to their event times.
"""
event_marks(h::History) = h.marks

"""
    event_marks(h, tmin, tmax)

Return the sorted vector of marks of events between `tmin` and `tmax` in `h`.
"""
function event_marks(h::History, tmin, tmax)
    @view h.marks[searchsortedfirst(h.times, tmin):(searchsortedfirst(h.times, tmax) - 1)]
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
    nb_events(h)

Count events in `h`.
"""
nb_events(h::History) = length(h.marks)

"""
    length(h)

Alias for `nb_events(h)`.
"""
Base.length(h::History) = nb_events(h)

"""
    nb_events(h, tmin, tmax)

Count events in `h` during the interval `[tmin, tmax)`.
"""
function nb_events(h::History, tmin, tmax)
    i_min = searchsortedfirst(event_times(h), tmin)
    i_max = searchsortedfirst(event_times(h), tmax)
    return i_max - i_min
end

"""
    has_events(h)

Check the presence of events in `h`.
"""
has_events(h::History) = nb_events(h) > 0

"""
    has_events(h, tmin, tmax)

Check the presence of events in `h` during the interval `[tmin, tmax)`.
"""
has_events(h::History, tmin, tmax) = nb_events(h, tmin, tmax) > 0

"""
    duration(h)

Compute the difference `h.tmax - h.tmin`.
"""
duration(h::History) = max_time(h) - min_time(h)

"""
    push!(h, t, m)

Add event `(t, m)` at the end of history `h`.
"""
function Base.push!(h::History, t::Real, m; check=true)
    if check
        @assert h.tmin <= t < h.tmax
        @assert (length(h) == 0) || (h.times[end] <= t)
    end
    push!(h.times, t)
    push!(h.marks, m)
    return nothing
end

"""
    append!(h, ts, ms)

Append events `(ts, ms)` at the end of history `h`.
"""
function Base.append!(h::History, ts::Vector{<:Real}, ms; check=true)
    if check
        perm = sortperm(ts)
        ts .= ts[perm]
        ms .= ms[perm]
        @assert h.tmin <= ts[1] && ts[end] < h.tmax
        @assert (length(h) == 0) || (h.times[end] <= ts[1])
    end
    append!(h.times, ts)
    append!(h.marks, ms)
    return nothing
end

"""
    cat(h1, h2)

If h1 and h2 are consecutive event histories, i.e., the end of
h1 coincides with the beginning of h2, then create a new event
history by concatenating h1 and h2.
"""
function Base.cat(h1::History, h2::History)
    max_time(h1) ≈ min_time(h2) || throw(
        DomainError(
            (h1.tmax, h2.tmin),
            "End of h1's interval must coincide with start of h2's interval",
        ),
    )
    times = [h1.times; h2.times]
    marks = [h1.marks; h2.marks]
    return History(; times=times, tmin=h1.tmin, tmax=h2.tmax, marks=marks, check=false)
end

"""
    time_change(h, Λ)

Apply the time rescaling `t -> Λ(t)` to history `h`.
"""
function time_change(h::History, Λ)
    new_times = Λ.(event_times(h))
    new_marks = copy(event_marks(h))
    new_tmin = Λ(min_time(h))
    new_tmax = Λ(max_time(h))
    return History(; times=new_times, marks=new_marks, tmin=new_tmin, tmax=new_tmax)
end

"""
    split_into_chunks(h, chunk_duration)

Split `h` into a vector of consecutive histories with individual duration `chunk_duration`.
"""
function split_into_chunks(h::History{T,M}, chunk_duration) where {T,M}
    chunks = History{T,M}[]
    limits = collect(min_time(h):chunk_duration:max_time(h))
    if !(limits[end] ≈ max_time(h))
        push!(limits, max_time(h))
    end
    for (a, b) in zip(limits[1:(end - 1)], limits[2:end])
        times = [t for t in event_times(h) if a <= t < b]
        marks = [m for (t, m) in zip(event_times(h), event_marks(h)) if a <= t < b]
        chunk = History(; times=times, marks=marks, tmin=a, tmax=b, check=false)
        push!(chunks, chunk)
    end
    return chunks
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
