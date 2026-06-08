@testset "Univariate History" begin
    # Constructors
    h_empty1 = History(0.0, 1.0)
    h_empty2 = History(0.0, 1.0, 2)

    @test h_empty1 isa History{Float64,Any}
    @test h_empty2 isa History{Float64,Any}
    @test ndims(h_empty1) == 1
    @test ndims(h_empty2) == 2
    @test isempty(h_empty1)
    @test isempty(h_empty2)

    h = History([0.2, 0.8, 1.1], 0.0, 2.0, ["a", "b", "c"]);

    @test h isa History{Float64,String}

    # Access
    @test duration(h) == 2.0
    @test nb_events(h) == 3
    @test nb_events(h, 1.0, 2.0) == 1
    @test has_events(h)
    @test !has_events(h, 1.5, 2.0)
    @test min_mark(h) == "a"
    @test max_mark(h) == "c"
    @test event_times(h) == [0.2, 0.8, 1.1]
    @test event_times(h, 0.2, 0.8) == [0.2]
    @test event_times(h, 0.8, 0.2) == []
    @test event_times(h, nothing) == h.times
    @test event_marks(h) == ["a", "b", "c"]
    @test event_marks(h, 0.2, 0.8) == ["a"]
    @test event_marks(h, 0.8, 0.2) == []
    @test ndims(h) == 1
    @test event_dims(h) == fill(nothing, 3)

    # Interface
    push!(h, 1.7, "d")

    @test has_events(h, 1.5, 2.0)

    h2 = History(; times=[2.3], marks=["e"], tmin=2.0, tmax=2.5)

    @test string(h2) == "History{Float64,String} with 1 events on interval [2.0, 2.5)"

    h_cat = cat(h, h2)

    @test nb_events(h_cat) == 5
    @test duration(h_cat) == 2.5
    @test length(split_into_chunks(h_cat, 0.3)) == 9

    h_exp = time_change(h_cat, exp)

    @test duration(h_exp) == exp(2.5) - exp(0.0)

    append!(h2, [2.4, 2.45], ["f", "g"])

    @test h2.times == [2.3, 2.4, 2.45]
    @test h2.marks == ["e", "f", "g"]

    @test isa(History(rand(3), 0, BigFloat(1)), History{BigFloat,Nothing})
    @test_throws DomainError History(rand(3), 1, 0)
    @test_throws DimensionMismatch History(rand(3), 0, 1, ["a", "b"])
    # Strict validation: events outside [tmin, tmax) must throw, not silently
    # discard. Same for unsorted times and same-time-same-dim repeats.
    @test_throws DomainError History([0.1, 1.1], 0, 1)
    @test_throws DomainError History([0.2, 0.1, 0.3], 0.0, 1.0)
    @test_throws DomainError History([0.1, 0.1], 0.0, 1.0)
end

@testset "Multivariate History" begin
    times1 = [0.1, 0.5]
    times2 = [0.2, 0.8]
    marks1 = ["a", "b"]
    marks2 = ["c", "d"]
    h_multi = History([times1, times2], 0.0, 1.0, [marks1, marks2])

    @test ndims(h_multi) == 2
    @test nb_events(h_multi) == 4
    @test event_times(h_multi) == [0.1, 0.2, 0.5, 0.8]
    @test event_marks(h_multi) == ["a", "c", "b", "d"]
    @test event_dims(h_multi) == [1, 2, 1, 2]

    h_multi1 = History(h_multi, 1)
    h_multi2 = History(h_multi, 2)

    @test event_times(h_multi1) == times1
    @test event_times(h_multi2) == times2
    @test event_marks(h_multi1) == marks1
    @test event_marks(h_multi2) == marks2

    @test_throws DomainError History(rand(3), 0, 1, rand(3), [1, 2, 3], 2)
    @test event_dims(History([[0.5]], 0, 1)) == [nothing]

    @test_throws DomainError History([1.0, 1.0, 2.0, 3.0, 4.0], 0.0, 5.0, fill(nothing, 5), [1, 1, 1, 2, 1], 2)
    @test_throws DomainError History([1.0, 1.0, 1.0, 2.0, 3.0], 0.0, 5.0, fill(nothing, 5), [1, 2, 1, 2, 1], 2)

    # Test dimension-specific methods
    @test event_times(h_multi, 1) == [0.1, 0.5]
    @test event_times(h_multi, 2) == [0.2, 0.8]
    @test event_marks(h_multi, 1) == ["a", "b"]
    @test event_marks(h_multi, 2) == ["c", "d"]
    @test nb_events(h_multi, 1) == 2
    @test nb_events(h_multi, 2) == 2

    # Test time and dimension specific
    @test event_times(h_multi, 0.0, 0.3, 1) == [0.1]
    @test event_marks(h_multi, 0.0, 0.3, 2) == ["c"]
    @test nb_events(h_multi, 0.0, 0.3, 1) == 1
    @test event_dims(h_multi, 0.0, 0.3) == [1, 2]

    # Test push! with dimension
    push!(h_multi, 0.85, "e", 1)
    @test nb_events(h_multi, 1) == 3
    @test event_marks(h_multi, 1) == ["a", "b", "e"]

    # Test append! with dimensions
    @test_throws AssertionError append!(h_multi, [0.8, 0.9], ["f", "g"], [2, 1])
    append!(h_multi, Float64[])
    append!(h_multi, [0.9, 0.95], ["g", "f"], [1, 2])
    @test nb_events(h_multi) == 7
    @test event_times(h_multi, 1) == [0.1, 0.5, 0.85, 0.9]
    @test event_marks(h_multi, 1) == ["a", "b", "e", "g"]
    # append! now validates strictly: unsorted input must throw instead of
    # being silently re-sorted (and silently mutating the caller's vectors).
    @test_throws AssertionError append!(h_multi, [0.99, 0.97], ["x", "y"], [1, 2])
end
