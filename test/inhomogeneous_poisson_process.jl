using DensityInterface
using Distributions
using ForwardDiff
using PointProcesses
using Statistics
using StatsAPI
using Test

rng = Random.seed!(42)

@testset verbose = true "InhomogeneousPoissonProcess" begin
    @testset "PolynomialIntensity" begin
        # Linear intensity: λ(t) = 1 + 0.5*t
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @testset "Intensity function evaluation" begin
            @test intensity_linear(0.0) ≈ 1.0
            @test intensity_linear(1.0) ≈ 1.5
            @test intensity_linear(2.0) ≈ 2.0
        end

        @testset "Interface methods" begin
            h_empty = History(Float64[], Float64[], 0.0, 10.0)
            @test ground_intensity(pp, 0.0, h_empty) ≈ 1.0
            @test ground_intensity(pp, 2.0, h_empty) ≈ 2.0
            @test mark_distribution(pp, 0.0, h_empty) isa Normal
            @test mark_distribution(pp) isa Normal
        end

        @testset "Simulation" begin
            h1 = rand(rng, pp, 0.0, 10.0)
            h2 = simulate_ogata(rng, pp, 0.0, 10.0)

            @test issorted(event_times(h1))
            @test issorted(event_times(h2))
            @test nb_events(h1) > 0
            @test nb_events(h2) > 0
            @test min_time(h1) == 0.0
            @test max_time(h1) == 10.0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], Float64[], 0.0, 10.0)
            # ∫₀¹⁰ (1 + 0.5*t) dt = [t + 0.25*t²]₀¹⁰ = 10 + 25 = 35
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 10.0)
            @test integral ≈ 35.0 rtol = 1e-3
        end

        @testset "Quadratic intensity" begin
            # λ(t) = 1 + 0.5*t + 0.1*t²
            intensity_quad = PolynomialIntensity([1.0, 0.5, 0.1])
            pp_quad = InhomogeneousPoissonProcess(intensity_quad, Categorical([0.3, 0.7]))

            @test intensity_quad(0.0) ≈ 1.0
            @test intensity_quad(1.0) ≈ 1.6
            @test intensity_quad(2.0) ≈ 2.4

            h = rand(rng, pp_quad, 0.0, 5.0)
            @test issorted(event_times(h))
            @test all(x -> x ∈ [1, 2], event_marks(h))
        end
    end

    @testset "ExponentialIntensity" begin
        # λ(t) = 2*exp(0.1*t)
        intensity_exp = ExponentialIntensity(2.0, 0.1)
        pp = InhomogeneousPoissonProcess(intensity_exp, Uniform())

        @testset "Intensity function evaluation" begin
            @test intensity_exp(0.0) ≈ 2.0
            @test intensity_exp(1.0) ≈ 2.0 * exp(0.1)
        end

        @testset "Simulation" begin
            h = rand(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], Float64[], 0.0, 10.0)
            # ∫ 2*exp(0.1*t) dt = 20*(exp(0.1*t))
            # From 0 to 10: 20*(exp(1) - exp(0)) = 20*(e - 1)
            expected = 20.0 * (exp(1.0) - 1.0)
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 10.0)
            @test integral ≈ expected rtol = 1e-3
        end
    end

    @testset "SinusoidalIntensity" begin
        # λ(t) = 5 + 2*sin(2π*t)
        intensity_sin = SinusoidalIntensity(5.0, 2.0, 2π, 0.0)
        pp = InhomogeneousPoissonProcess(intensity_sin, Normal())

        @testset "Intensity function evaluation" begin
            @test intensity_sin(0.0) ≈ 5.0
            @test intensity_sin(0.25) ≈ 7.0 rtol = 1e-6
            @test intensity_sin(0.5) ≈ 5.0 rtol = 1e-6
            @test intensity_sin(0.75) ≈ 3.0 rtol = 1e-6
            @test intensity_sin(1.0) ≈ 5.0 rtol = 1e-6
        end

        @testset "Simulation" begin
            h = rand(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], Float64[], 0.0, 1.0)
            # Over one period, sin integrates to 0, so integral = 5*1 = 5
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 1.0)
            @test integral ≈ 5.0 rtol = 1e-3
        end

        @testset "Intensity bounds" begin
            h_empty = History(Float64[], Float64[], 0.0, 10.0)
            B, L = ground_intensity_bound(pp, 0.0, h_empty)
            @test B >= 7.0  # max is a + |b| = 5 + 2 = 7
            @test L == typemax(Float64)  # Bound holds for all time
        end
    end

    @testset "PiecewiseConstantIntensity" begin
        breakpoints = [0.0, 1.0, 2.0, 3.0]
        rates = [1.0, 3.0, 2.0]
        intensity_pw = PiecewiseConstantIntensity(breakpoints, rates)
        pp = InhomogeneousPoissonProcess(intensity_pw, Categorical([0.5, 0.5]))

        @testset "Intensity function evaluation" begin
            @test intensity_pw(0.5) ≈ 1.0
            @test intensity_pw(1.5) ≈ 3.0
            @test intensity_pw(2.5) ≈ 2.0
        end

        @testset "Simulation" begin
            h = rand(rng, pp, 0.0, 3.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], Float64[], 0.0, 3.0)
            # ∫₀³ = 1*(1-0) + 3*(2-1) + 2*(3-2) = 1 + 3 + 2 = 6
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 3.0)
            @test integral ≈ 6.0 rtol = 1e-3
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError PiecewiseConstantIntensity([0.0, 1.0], [1.0, 2.0])
            @test_throws ArgumentError PiecewiseConstantIntensity([1.0, 0.0], [1.0])
            @test_throws ArgumentError PiecewiseConstantIntensity([0.0, 1.0], [-1.0])
        end
    end

    @testset "LinearCovariateIntensity" begin
        # λ(t) = 1.0 + 0.5*t + 0.3*sin(2π*t)
        cov1 = t -> t
        cov2 = t -> sin(2π * t)
        intensity_cov = LinearCovariateIntensity(1.0, [0.5, 0.3], [cov1, cov2])
        pp = InhomogeneousPoissonProcess(intensity_cov, Normal())

        @testset "Intensity function evaluation" begin
            @test intensity_cov(0.0) ≈ 1.0
            @test intensity_cov(1.0) ≈ 1.5 rtol = 1e-6
            @test intensity_cov(0.25) ≈ 1.0 + 0.5 * 0.25 + 0.3 * 1.0 rtol = 1e-6
        end

        @testset "Simulation" begin
            h = rand(rng, pp, 0.0, 5.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError LinearCovariateIntensity(1.0, [0.5], [cov1, cov2])
        end
    end

    @testset "Custom intensity function" begin
        # User-defined function
        custom_func = t -> 1.0 + 0.5 * sin(t)^2
        pp = InhomogeneousPoissonProcess(custom_func, Uniform())

        @testset "Simulation" begin
            h = rand(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Interface methods" begin
            h_empty = History(Float64[], Float64[], 0.0, 10.0)
            @test ground_intensity(pp, 0.0, h_empty) ≈ 1.0
            @test ground_intensity(pp, π / 2, h_empty) ≈ 1.5
        end
    end

    @testset "Fitting - PiecewiseConstant" begin
        # Generate data from known piecewise constant process
        breakpoints_true = [0.0, 5.0, 10.0]
        rates_true = [2.0, 5.0]
        intensity_true = PiecewiseConstantIntensity(breakpoints_true, rates_true)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Normal())

        h = rand(rng, pp_true, 0.0, 10.0)

        # Fit with 2 bins
        pp_est = fit(
            InhomogeneousPoissonProcess{PiecewiseConstantIntensity{Float64},Normal}, h, 2
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa PiecewiseConstantIntensity
        @test length(pp_est.intensity_function.rates) == 2

        # Rates should be roughly similar to true rates
        est_rates = pp_est.intensity_function.rates
        @test 0.5 * rates_true[1] <= est_rates[1] <= 2.0 * rates_true[1]
        @test 0.5 * rates_true[2] <= est_rates[2] <= 2.0 * rates_true[2]
    end

    @testset "Fitting - Polynomial" begin
        # Generate data from linear process
        intensity_true = PolynomialIntensity([2.0, 0.3])
        pp_true = InhomogeneousPoissonProcess(intensity_true, Categorical([0.4, 0.6]))

        h = rand(rng, pp_true, 0.0, 20.0)

        # Fit linear intensity
        pp_est = fit(
            InhomogeneousPoissonProcess{PolynomialIntensity{Float64},Categorical}, h, 1
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa PolynomialIntensity
        @test pp_est.mark_dist isa Categorical

        # Check mark distribution is reasonable
        @test length(pp_est.mark_dist.p) == 2
    end

    @testset "Log-density and gradients" begin
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        h = rand(rng, pp, 0.0, 10.0)
        l = logdensityof(pp, h)

        @test l isa Real
        @test isfinite(l)

        # Test ForwardDiff compatibility
        f(a) = begin
            pp_test = InhomogeneousPoissonProcess(PolynomialIntensity([a, 0.5]), Normal())
            return logdensityof(pp_test, h)
        end

        g = ForwardDiff.derivative(f, 1.0)
        @test isfinite(g)
    end

    @testset "DensityInterface" begin
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @test DensityKind(pp) == HasDensity()
    end

    @testset "Display methods" begin
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @test string(intensity_linear) == "PolynomialIntensity([1.0, 0.5])"
        @test occursin("InhomogeneousPoissonProcess", string(pp))
    end
end
