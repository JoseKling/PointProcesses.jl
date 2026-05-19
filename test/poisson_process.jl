rng = Random.seed!(63)

@testset "Constructors" begin
    pp1 = PoissonProcess(1.0, Normal())
    pp2 = PoissonProcess(1.0)
    pp3 = PoissonProcess()

    @test ndims(pp1) == 1
    @test pp1 isa PoissonProcess
    @test pp2 isa PoissonProcess
    @test pp3 isa PoissonProcess
    @test pp1.λ == 1.0
    @test pp2.λ == 1.0
    @test pp3.λ == 1.0
    @test pp1.mark_dist == Normal()
    @test pp2.mark_dist == Dirac(nothing)
    @test pp3.mark_dist == Dirac(nothing)
    @test_throws DomainError PoissonProcess(-1.0)
end

pp = PoissonProcess(1.0, Normal())

@testset "Interface" begin
    @test string(pp) == "PoissonProcess(1.0, Normal{Float64})"
    @test DensityKind(pp) == HasDensity()

    h = History([0.1, 0.5], 0, 1, [0.0, 0.0])

    @test ground_intensity(pp, 0, h) == 1.0
    @test mark_distribution(pp) == Normal()
    @test intensity(pp, 0.5, 0, h) == 1.0 * pdf(Normal(), 0.5)
    @test log_intensity(pp, 0.5, 0, h) == log(1.0) + logpdf(Normal(), 0.5)
    @test integrated_ground_intensity(pp, h, 0.0, 1.0) == 1.0
    @test ground_intensity_bound(pp, 0.0, h) == (1.0, Inf)
    @test logdensityof(pp, h) ≈ 2 * log(pdf(Normal(), 0.0)) - 1.0

    f2(λ) = logdensityof(PoissonProcess(λ, Normal()), h)
    gf = ForwardDiff.derivative(f2, 3.0)
    @test all(gf .< 0)
end

pp0 = PoissonProcess(0.0, Normal())
ppvec = PoissonProcess(1.0, MvNormal(Matrix(I, 2, 2)))

@testset "Simulation" begin
    h1 = simulate(rng, pp, 0.0, 1000.0)
    h2 = simulate_ogata(rng, pp, 0.0, 1000.0)
    h3 = simulate(rng, pp0, 0.0, 1000.0)
    h4 = simulate_ogata(rng, pp0, 0.0, 1000.0)

    @test issorted(event_times(h1))
    @test issorted(event_times(h2))
    @test nb_events(h1) > 0
    @test nb_events(h2) > 0
    @test !has_events(h3)
    @test !has_events(h4)
end

@testset "Fitting" begin
    h1 = simulate(rng, pp, 0.0, 1000.0)
    h2 = simulate_ogata(rng, pp, 0.0, 1000.0)

    pp_est1 = fit(PoissonProcess{Float32,Normal}, [h1, h1])
    pp_est2 = fit(PoissonProcess{Float32,Normal}, [h2, h2])

    λ_error1 = mean(abs, pp_est1.λ - pp.λ)
    λ_error2 = mean(abs, pp_est2.λ - pp.λ)
    μ_error1 = mean(abs, pp_est1.mark_dist.μ - pp.mark_dist.μ)
    σ_error1 = mean(abs, pp_est1.mark_dist.σ - pp.mark_dist.σ)
    μ_error2 = mean(abs, pp_est2.mark_dist.μ - pp.mark_dist.μ)
    σ_error2 = mean(abs, pp_est2.mark_dist.σ - pp.mark_dist.σ)

    @test λ_error1 < 0.1
    @test λ_error2 < 0.1
    @test μ_error1 < 0.1
    @test σ_error1 < 0.1
    @test μ_error2 < 0.1
    @test σ_error2 < 0.1

    pp_est1 = fit(PoissonProcess{Float32,Normal}, h1)
    l = logdensityof(pp, h1)
    l_est = logdensityof(pp_est1, h1)
    @test l_est > l
end

@testset "Type promotion in ground_intensity_bound" begin
    pp32 = PoissonProcess(1.0f0)
    h32 = History(Float32[], 0.0f0, 1.0f0)
    tup = ground_intensity_bound(pp32, 0.0f0, h32)
    @test typeof(tup[1]) === typeof(tup[2]) === Float32   # Float32 preserved
    @test tup[2] === typemax(Float32)

    # Mixed: Float32 λ + Float64 t → both Float64
    h64 = History(Float64[], 0.0, 1.0)
    tup = ground_intensity_bound(pp32, 0.0, h64)
    @test typeof(tup[1]) === typeof(tup[2]) === Float64
end
