using Test
using PointProcesses
using Distributions
using Random
using ForwardDiff

@testset "PoissonProcess" begin
    @testset "Constructors" begin
        # Test basic constructor
        pp = PoissonProcess([1.0], [Dirac(nothing)])
        @test pp.λ == [1.0]
        @test pp.mark_dist == [Dirac(nothing)]
        @test ndims(pp) == 1

        # Test with multiple dimensions
        pp_multi = PoissonProcess([1.0, 2.0], [Dirac(nothing), Normal(0,1)])
        @test pp_multi.λ == [1.0, 2.0]
        @test length(pp_multi.mark_dist) == 2
        @test ndims(pp_multi) == 2

        # Test default constructor
        pp_default = PoissonProcess()
        @test pp_default.λ == [1.0]
        @test pp_default.mark_dist == [Dirac(nothing)]

        # Test univariate unmarked process
        uupp = PoissonProcess([1.0])
        @test uupp isa PoissonProcess
        @test uupp.mark_dist == [Dirac(nothing)]
        
        # Test univariate marked process
        umpp= PoissonProcess(1.0, Normal(0,1))
        @test umpp.λ == [1.0]
        @test umpp.mark_dist == [Normal(0,1)]

        # Test multivariate with equal marks
        mepp = PoissonProcess([1.0, 1.0], Normal(0,1))
        @test mepp isa PoissonProcess
        @test mepp.mark_dist == [Normal(0,1), Normal(0,1)]
        @test ndims(mepp) == 2

        # Test domain error for negative λ
        @test_throws DomainError PoissonProcess([-1.0, 1.0])
    end

    @testset "Show" begin
        pp = PoissonProcess([1.0], [Normal])
        str = sprint(show, pp)
        @test occursin("Univariate PoissonProcess", str)

        upp = PoissonProcess()
        str_upp = sprint(show, upp)
        @test occursin("Univariate Unmarked PoissonProcess", str_upp)

        # Multivariate unmarked
        pp_multi_unmarked = PoissonProcess([1.0, 2.0])
        str_multi_unmarked = sprint(show, pp_multi_unmarked)
        @test occursin("2-dimensional UnmarkedPoissonProcess", str_multi_unmarked)

        # Multivariate marked
        pp_multi_marked = PoissonProcess([1.0, 0.5], [Normal(), Normal(1.0, 2.0)])
        str_multi_marked = sprint(show, pp_multi_marked)
        @test occursin("2-dimensional PoissonProcess", str_multi_marked)
    end

    @testset "Access Methods" begin
        pp = PoissonProcess([1.0, 2.0], [Dirac(nothing), Normal(0,1)])
        h = History(rand(2), 0, 1)

        @test ground_intensity(pp, 0, h) == [1.0, 2.0]
        @test ground_intensity(pp, 0, h, 1) == 1.0
        @test mark_distribution(pp, 0) == [Dirac(nothing), Normal(0,1)]
        @test mark_distribution(pp, 0, h, 2) == Normal(0,1)
    end

    @testset "Intensity Functions" begin
        pp = PoissonProcess([1.0, 2.0], [Normal(), Normal()])
        h = History(Float64[], 0, 1)

        @test intensity(pp, 0.0, 1, h) ≈ [pdf(Normal(), 0.0), 2 * pdf(Normal(), 0.0)]
        @test intensity(pp, 0.0, 0, h, 1) ≈ pdf(Normal(), 0.0)
        @test log_intensity(pp, 0.0, 0, h) ≈ log.([pdf(Normal(), 0.0), 2 * pdf(Normal(), 0.0)])
        @test log_intensity(pp, 0.0, 0, h, 1) ≈ log(pdf(Normal(), 0.0))

        BLs = ground_intensity_bound(pp, 1.0, h)
        @test length(BLs) == 2
        @test BLs[1] == (1.0, Inf)
        @test BLs[2] == (2.0, Inf)

        B_d, L_d = ground_intensity_bound(pp, 1.0, h, 2)
        @test B_d == 2.0
        @test L_d == Inf

        @test integrated_ground_intensity(pp, h, 0.0, 1.0) == [1.0, 2.0]
        @test integrated_ground_intensity(pp, h, 0.0, 1.0, 1) == 1.0
    end

    @testset "Time Change" begin
        pp = PoissonProcess()
        h = History([0.5, 1.0], 0.0, 2.0)
        tc = time_change(h, pp)
        @test tc.tmin == 0.0
        @test tc.tmax == 2.0
        @test length(tc.times) == 2
    end

    @testset "Simulation" begin
        rng = MersenneTwister(42)
        pp = PoissonProcess(100.0)
        h = simulate(rng, pp, 0.0, 1.0)
        @test h.tmin == 0.0
        @test h.tmax == 1.0
        @test issorted(h.times)
        @test all(0.0 .<= h.times .<= 1.0)
        @test nb_events(h) > 0
        @test nb_events(h, 1) > 0

        # Multivariate simulation
        pp_multi = PoissonProcess([50.0, 200.0])
        h_multi = simulate(rng, pp_multi, 0.0, 1.0)
        @test ndims(h_multi) == 2
        @test nb_events(h_multi) > 0
        @test nb_events(h_multi, 1) >= 0
        @test nb_events(h_multi, 2) > nb_events(h_multi, 1)
    end

    @testset "Fitting" begin
        h = History([0.1, 0.5, 0.9], 0.0, 1.0)
        pp_fit = fit(PoissonProcess{Float64, Dirac{Nothing}}, h)
        @test pp_fit.λ ≈ [3.0]

        # Fit with multiple histories
        pp_fit_multi = fit(PoissonProcess{Float64, Dirac{Nothing}}, [h, h])
        @test pp_fit_multi.λ ≈ [3.0]

        # With marks
        h_marked = History([0.1, 0.5, 0.9], 0.0, 1.0, [1.0, 2.0, 3.0])
        pp_marked = fit(PoissonProcess{Float64, Normal}, h_marked)
        @test length(pp_marked.λ) == 1
        @test pp_marked.λ[1] ≈ 3.0

        # Fit marked with multiple histories
        pp_marked_multi = fit(PoissonProcess{Float64, Normal}, [h_marked, h_marked])
        @test length(pp_marked_multi.λ) == 1
        @test pp_marked_multi.λ[1] ≈ 3.0

        # Fit with prior
        prior = PoissonProcessPrior([1.0], 1.0)
        pp_map = fit_map(UnmarkedPoissonProcess{Float64}, prior, [h, h])
        @test length(pp_map.λ) == 1
    end

    @testset "COmpatibility with ForwardDiff" begin
        h_1d = History([0.1, 0.5, 0.9], 0.0, 1.0, rand(3))
        f1(λ) = logdensityof(PoissonProcess([λ]), h_unmarked)
        df = ForwardDiff.derivative(f1, 30.0)
        @test df <: Real
        h_2d = History(rand(6), 0.0, 1.0, rand(6), [1, 1, 1, 2, 2, 2])
        f2(λ) = sum(logdensityof(PoissonProcess(λ), h_2d, 1))
        gf = ForwardDiff.gradient(f2, [30.0, 10.0])
        @test eltype(gf) <: Real
    end

    @testset "Sufficient Statistics" begin
        h1 = History([0.1, 0.5], 0.0, 1.0)
        h2 = History([0.2, 0.8], 0.0, 1.0)
        ss = suffstats(PoissonProcess, [h1, h2])
        @test ss.nb_events == [4]
        @test ss.duration == 2.0
    end

    @testset "Prior" begin
        prior = PoissonProcessPrior([1.0], 1.0)
        pp = PoissonProcess([2.0], [Dirac(nothing)])
        ld = logdensityof(prior, pp)
        @test ld isa Real
    end
end
