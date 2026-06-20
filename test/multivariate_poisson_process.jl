rng = Random.seed!(63)
λ = [1.0, 2.0, 3.0]
mark_dists = [Normal(), Exponential(1.0), Uniform(0, 1)]

pp1 = PoissonProcess(λ, mark_dists)
pp2 = PoissonProcess(λ)
pp3 = PoissonProcess(λ, Normal())

@testset "Constructors" begin
    @test pp1 isa MultivariatePoissonProcess
    @test pp2 isa MultivariatePoissonProcess
    @test pp3 isa MultivariatePoissonProcess
end

@testset "Interface" begin
    h = History([[0.1, 0.7], [0.3], [0.5]], 0, 1, [[0.0, 0.0], [0.0], [0.0]])

    @test contains(string(pp1), "MultivariatePoissonProcess")
    @test DensityKind(pp) == HasDensity()

    @test ndims(pp1) == 3
    @test ndims(pp2) == 3
    @test ndims(pp3) == 3

    @test ground_intensity(pp1, 0, h) == λ
    @test ground_intensity(pp1, 0, h, 2) == λ[2]
    @test mark_distribution(pp1, 0, h) == mark_dists
    @test mark_distribution(pp1, 0, h, 2) == mark_dists[2]
    @test intensity(pp1, 0, 0, h) ==
        [intensity(pp1.processes[d], 0, 0, h) for d in 1:ndims(pp1)]
    @test intensity(pp1, 0, 0, h, 1) == intensity(pp1.processes[1], 0, 0, h)
    @test log_intensity(pp1, 0, 0, h) ≈
        [log_intensity(pp1.processes[d], 0, 0, h) for d in 1:ndims(pp1)]
    @test log_intensity(pp1, 0, 0, h, 1) ≈ log_intensity(pp1.processes[1], 0, 0, h)
    @test ground_intensity_bound(pp1, 0.0, h) == [(1.0, Inf), (2.0, Inf), (3.0, Inf)]
    @test ground_intensity_bound(pp1, 0.0, h, 1) == (1.0, Inf)
    @test integrated_ground_intensity(pp1, h, 0, 1) == [1.0, 2.0, 3.0]
    @test integrated_ground_intensity(pp1, h, 0, 1, 1) == 1.0

    h1 = simulate(rng, pp2, 0.0, 1000.0)

    f1(λ) = logdensityof(PoissonProcess(λ), h1)
    gf = ForwardDiff.gradient(f1, 10 * ones(10))

    @test all(gf .< 0)
end

@testset "Simulation" begin
    pp0 = PoissonProcess([0.0, 1.0, 0.0])
    bpp = BoundedPointProcess(pp1, 0.0, 1000.0)
    h1 = simulate(rng, pp1, 0.0, 1000.0)
    h2 = simulate(rng, pp0, 0.0, 1000.0)
    h3 = simulate(rng, bpp)

    @test nb_events(h1) > 0
    @test nb_events(h2) > 0
    @test nb_events(h3) > 0
    @test ndims(h1) == 3
    @test ndims(h2) == 3
    @test ndims(h3) == 3
    @test issorted(event_times(h1))
    @test issorted(event_times(h2))

    @test all(event_marks(h2) .== nothing)
    @test isempty(event_times(h2, 1))
    @test isempty(event_times(h2, 3))
end

@testset "Fitting" begin
    h1 = simulate(rng, pp2, 0.0, 1000.0)
    h2 = History(fill(rand(100) .* 1000, 3), 0.0, 1000.0)

    pp_est1 = fit(fill(PoissonProcess{Float64,NoMarks}, 3), h1)
    pp_est2 = fit(fill(PoissonProcess{Float64,NoMarks}, 3), h2)
    λ_est1 = [pp_est1.processes[d].λ for d in 1:ndims(pp_est1)]
    λ_est2 = [pp_est2.processes[d].λ for d in 1:ndims(pp_est2)]

    λ_error1 = mean(abs, λ_est1 - λ)
    λ_error2 = mean(abs, λ_est2 - λ)

    l = logdensityof(pp2, h1)
    l_est = logdensityof(pp_est1, h1)

    @test λ_error1 < λ_error2
    @test l_est > l

    pp_marks = PoissonProcess([1, 1], Normal())
    h_marks = simulate(pp_marks, 0.0, 1000.0)
    pp_est_marks = fit(fill(PoissonProcess{Float64,Normal}, 2), h_marks)
    @test pp_est_marks isa MultivariatePoissonProcess
    @test pp_est_marks.processes[1].mark_dist isa Normal
    @test logdensityof(pp_est_marks, h_marks) >= logdensityof(pp_marks, h_marks)
end

@testset "Time change" begin
    h = History([[1.7], [1.3, 1.4], [1.1, 1.5, 1.9]], 1.0, 2.0)
    h_transf = time_change(h, pp2)

    @test h_transf.tmin == 0.0
    @test h_transf.tmax == 3.0
    @test nb_events(h_transf) == nb_events(h)
    @test all(event_times(h_transf) .≈ sort([0.7, 0.6, 0.8, 0.3, 1.5, 2.7]))
    @test event_dims(h_transf) == [3, 2, 1, 2, 3, 3]
end

@testset "MultivariatePoissonProcessPrior" begin
    α = [1.0, 2.0, 3.0]
    β = 0.5

    h = simulate(pp2, 0.0, 1000.0)
    prior = MultivariatePoissonProcessPrior(α, β)
    pp_est = fit_map(MultivariatePoissonProcess, prior, h)

    l = logdensityof(prior, pp_est)

    @test l isa Real
    @test isfinite(l)
end
