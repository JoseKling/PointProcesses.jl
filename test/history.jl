h = History([0.2, 0.8, 1.1], 0.0, 2.0, ["a", "b", "c"]);

@test duration(h) == 2.0
@test nb_events(h) == 3
@test nb_events(h, 1.0, 2.0) == 1
@test has_events(h)
@test !has_events(h, 1.5, 2.0)
@test min_mark(h) == "a"
@test max_mark(h) == "c"
@test event_times(h, 0.2, 0.8) == [0.2]
@test event_marks(h, 0.2, 0.8) == ["a"]

push!(h, 1.7, "d")

@test has_events(h, 1.5, 2.0)

h2 = union(h1, History([2.3], 2, 2.5, ["e"]))

@test string(h2) == "History{String,Float64} with 1 events on interval [2.0, 2.5)"
@test nb_events(h2) == 5

h_exp = time_change(h2, exp)

@test duration(h_exp) == exp(2.5) - exp(0.0)

@test length(split_into_chunks(h2, 0.3)) == 9
