using DensityInterface
using Distributions
using ForwardDiff
using Optim
using PointProcesses
using Statistics
using StatsAPI
using Test

rng = Random.seed!(12345)

@testset verbose = true "InhomogeneousPoissonProcess" begin
    @testset "PolynomialIntensity" begin
        # Linear intensity: λ(t) = 1 + 0.5*t
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @testset "Intensity function evaluation - identity link" begin
            @test intensity_linear(0.0) ≈ 1.0
            @test intensity_linear(1.0) ≈ 1.5
            @test intensity_linear(2.0) ≈ 2.0
        end

        @testset "Intensity function evaluation - log link" begin
            # Linear log: λ(t) = exp(0.5 + 0.1*t)
            intensity_log = PolynomialIntensity([0.5, 0.1]; link=:log)
            @test intensity_log(0.0) ≈ exp(0.5)
            @test intensity_log(1.0) ≈ exp(0.6)
            @test intensity_log(2.0) ≈ exp(0.7)
            # Verify positivity
            @test all(intensity_log(t) > 0 for t in 0.0:0.1:10.0)
        end

        @testset "Link function validation" begin
            @test_throws ArgumentError PolynomialIntensity([1.0, 0.5]; link=:invalid)
        end

        @testset "Interface methods" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            @test ground_intensity(pp, 0.0, h_empty) ≈ 1.0
            @test ground_intensity(pp, 2.0, h_empty) ≈ 2.0
            @test mark_distribution(pp, 0.0, h_empty) isa Normal
            @test mark_distribution(pp) isa Normal
        end

        @testset "Simulation" begin
            h1 = simulate(rng, pp, 0.0, 10.0)
            h2 = simulate_ogata(rng, pp, 0.0, 10.0)

            @test issorted(event_times(h1))
            @test issorted(event_times(h2))
            @test nb_events(h1) > 0
            @test nb_events(h2) > 0
            @test min_time(h1) == 0.0
            @test max_time(h1) == 10.0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
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

            h = simulate(rng, pp_quad, 0.0, 5.0)
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
            # Verify positivity for all t
            @test all(intensity_exp(t) > 0 for t in -10.0:0.1:10.0)
        end

        @testset "Positivity enforcement" begin
            @test_throws ArgumentError ExponentialIntensity(-1.0, 0.1)
            @test_throws ArgumentError ExponentialIntensity(0.0, 0.1)
            # Negative b is fine (decreasing intensity)
            @test ExponentialIntensity(2.0, -0.1) isa ExponentialIntensity
        end

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
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
            # Verify positivity
            @test all(intensity_sin(t) >= 0 for t in 0.0:0.01:10.0)
        end

        @testset "Positivity enforcement" begin
            # Valid: a >= |b|
            @test SinusoidalIntensity(5.0, 3.0, 2π) isa SinusoidalIntensity
            @test SinusoidalIntensity(5.0, -3.0, 2π) isa SinusoidalIntensity
            @test SinusoidalIntensity(3.0, 3.0, 2π) isa SinusoidalIntensity

            # Invalid: a < |b|
            @test_throws ArgumentError SinusoidalIntensity(2.0, 3.0, 2π)
            @test_throws ArgumentError SinusoidalIntensity(2.0, -3.0, 2π)
        end

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 1.0, Float64[])
            # Over one period, sin integrates to 0, so integral = 5*1 = 5
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 1.0)
            @test integral ≈ 5.0 rtol = 1e-3
        end

        @testset "Integrated intensity with ω ≈ 0" begin
            # When ω ≈ 0, sin(ω*t + φ) ≈ sin(φ) (approximately constant)
            intensity_zero_omega = SinusoidalIntensity(5.0, 2.0, 1e-12, π / 4)
            pp_zero_omega = InhomogeneousPoissonProcess(intensity_zero_omega, Uniform())
            h_empty = History(Float64[], 0.0, 10.0, Float64[])

            # ∫₀¹⁰ (5 + 2*sin(φ)) dt ≈ (5 + 2*sin(π/4)) * 10
            expected = (5.0 + 2.0 * sin(π / 4)) * 10.0
            integral = integrated_ground_intensity(pp_zero_omega, h_empty, 0.0, 10.0)
            @test integral ≈ expected rtol = 1e-3
        end

        @testset "Intensity bounds" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
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
            h = simulate(rng, pp, 0.0, 3.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 3.0, Float64[])
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
        # λ(t) = exp(1.0 + 0.5*t + 0.3*sin(2π*t))
        cov1 = t -> t
        cov2 = t -> sin(2π * t)
        intensity_cov = LinearCovariateIntensity(1.0, [0.5, 0.3], [cov1, cov2])
        pp = InhomogeneousPoissonProcess(intensity_cov, Normal())

        @testset "Intensity function evaluation" begin
            # η(t) = 1.0 + 0.5*t + 0.3*sin(2π*t)
            # λ(t) = exp(η(t))
            @test intensity_cov(0.0) ≈ exp(1.0) rtol = 1e-6

            @test intensity_cov(1.0) ≈ exp(1.0 + 0.5*1.0) rtol = 1e-6

            @test intensity_cov(0.25) ≈ exp(1.0 + 0.5*0.25 + 0.3*1.0) rtol = 1e-6
        end

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 5.0)
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
            h = simulate(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Interface methods" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
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

        h = simulate(rng, pp_true, 0.0, 10.0)

        # Fit with 2 bins
        pp_est = fit(
            InhomogeneousPoissonProcess{
                PiecewiseConstantIntensity{Float64},Normal,IntegrationConfig
            },
            h,
            2,
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa PiecewiseConstantIntensity
        @test length(pp_est.intensity_function.rates) == 2

        # Rates should be roughly similar to true rates
        est_rates = pp_est.intensity_function.rates
        @test 0.5 * rates_true[1] <= est_rates[1] <= 2.0 * rates_true[1]
        @test 0.5 * rates_true[2] <= est_rates[2] <= 2.0 * rates_true[2]
    end

    @testset "Fitting - Multiple Histories" begin
        @testset "PiecewiseConstant with multiple histories" begin
            # Test that fit correctly concatenates multiple histories
            # NOTE: The fit function concatenates histories by combining event times
            # and using min/max of time bounds

            # Create 3 histories with non-overlapping time ranges and known properties
            breakpoints1 = [0.0, 10.0]
            rates1 = [3.0]
            intensity1 = PiecewiseConstantIntensity(breakpoints1, rates1)
            pp1 = InhomogeneousPoissonProcess(intensity1, Normal())
            h1 = simulate(rng, pp1, 0.0, 10.0)

            breakpoints2 = [10.0, 20.0]
            rates2 = [5.0]
            intensity2 = PiecewiseConstantIntensity(breakpoints2, rates2)
            pp2 = InhomogeneousPoissonProcess(intensity2, Normal())
            h2 = simulate(rng, pp2, 10.0, 20.0)

            breakpoints3 = [20.0, 30.0]
            rates3 = [2.0]
            intensity3 = PiecewiseConstantIntensity(breakpoints3, rates3)
            pp3 = InhomogeneousPoissonProcess(intensity3, Normal())
            h3 = simulate(rng, pp3, 20.0, 30.0)

            histories = [h1, h2, h3]

            # Fit using 3 bins (one for each original history)
            pp_est = fit(
                InhomogeneousPoissonProcess{
                    PiecewiseConstantIntensity{Float64},Normal,IntegrationConfig
                },
                histories,
                3,
            )

            @test pp_est isa InhomogeneousPoissonProcess
            @test pp_est.intensity_function isa PiecewiseConstantIntensity
            @test length(pp_est.intensity_function.rates) == 3

            # Check that the combined time range is correct
            @test pp_est.intensity_function.breakpoints[1] == 0.0
            @test pp_est.intensity_function.breakpoints[end] == 30.0

            # Check that total events match
            total_events = sum(nb_events(h) for h in histories)
            @test total_events == nb_events(h1) + nb_events(h2) + nb_events(h3)
            @test total_events > 0
        end

        @testset "Exponential with multiple histories" begin
            # Create multiple histories from exponential process
            # NOTE: Using non-overlapping time ranges since fit concatenates
            intensity_true = ExponentialIntensity(2.0, 0.05)
            pp_true = InhomogeneousPoissonProcess(intensity_true, Uniform())

            # Generate 4 histories with non-overlapping time ranges
            histories = [
                simulate(rng, pp_true, 0.0, 25.0),
                simulate(rng, pp_true, 25.0, 50.0),
                simulate(rng, pp_true, 50.0, 75.0),
                simulate(rng, pp_true, 75.0, 100.0),
            ]

            # Fit using multiple histories (they will be concatenated)
            pp_est = fit(
                InhomogeneousPoissonProcess{
                    ExponentialIntensity{Float64},Uniform,IntegrationConfig
                },
                histories,
                [log(2.0), 0.05],
            )

            @test pp_est isa InhomogeneousPoissonProcess
            @test pp_est.intensity_function isa ExponentialIntensity
            @test pp_est.mark_dist isa Uniform

            # With concatenated non-overlapping data, parameters should be in reasonable range
            @test 0.5 * intensity_true.a <=
                pp_est.intensity_function.a <=
                3.0 * intensity_true.a
            @test abs(pp_est.intensity_function.b - intensity_true.b) < 0.15
        end

        @testset "Multiple histories with different time ranges" begin
            # Test that fit function correctly handles min/max when combining histories
            # with different (but non-overlapping) time ranges
            intensity_true = PolynomialIntensity([2.0, 0.1])
            pp_true = InhomogeneousPoissonProcess(intensity_true, Categorical([0.5, 0.5]))

            # Generate 4 histories with different non-overlapping time ranges
            histories = [
                simulate(rng, pp_true, 0.0, 25.0),
                simulate(rng, pp_true, 30.0, 55.0),
                simulate(rng, pp_true, 60.0, 85.0),
                simulate(rng, pp_true, 90.0, 115.0),
            ]

            # Fit using all histories - should use min/max across all ranges
            pp_est = fit(
                InhomogeneousPoissonProcess{
                    PolynomialIntensity{Float64},Categorical,IntegrationConfig
                },
                histories,
                [2.0, 0.1],
            )

            @test pp_est isa InhomogeneousPoissonProcess
            @test pp_est.intensity_function isa PolynomialIntensity

            # Parameters should be estimated reasonably
            @test abs(pp_est.intensity_function.coefficients[1] - 2.0) < 1.5
            @test abs(pp_est.intensity_function.coefficients[2] - 0.1) < 0.3
        end

        @testset "Single history in vector" begin
            # Test that passing a single history in a vector still works
            intensity_true = ExponentialIntensity(3.0, 0.1)
            pp_true = InhomogeneousPoissonProcess(intensity_true, Normal())

            h = simulate(rng, pp_true, 0.0, 15.0)
            histories = [h]

            pp_est = fit(
                InhomogeneousPoissonProcess{
                    ExponentialIntensity{Float64},Normal,IntegrationConfig
                },
                histories,
                [log(3.0), 0.1],
            )

            @test pp_est isa InhomogeneousPoissonProcess
            @test pp_est.intensity_function isa ExponentialIntensity
        end
    end

    @testset "Fitting - Polynomial (MLE)" begin
        # Generate data from linear process with log link
        intensity_true = PolynomialIntensity([0.5, 0.1]; link=:log)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Categorical([0.4, 0.6]))

        h = simulate(rng, pp_true, 0.0, 50.0)

        # Fit linear intensity with MLE
        # Initial params for degree 1 polynomial: [a0, a1]
        pp_est = fit(
            InhomogeneousPoissonProcess{
                PolynomialIntensity{Float64},Categorical,IntegrationConfig
            },
            h,
            [0.5, 0.1];
            link=:log,
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa PolynomialIntensity
        @test pp_est.mark_dist isa Categorical
        @test pp_est.intensity_function.link === :log

        # Check mark distribution is reasonable
        @test length(pp_est.mark_dist.p) == 2

        # Parameters should be in the right ballpark
        @test abs(pp_est.intensity_function.coefficients[1] - 0.5) < 0.5
        @test abs(pp_est.intensity_function.coefficients[2] - 0.1) < 0.2
    end

    @testset "Fitting - Exponential (MLE)" begin
        # Generate data from exponential process
        intensity_true = ExponentialIntensity(2.0, 0.05)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Normal())

        h = simulate(rng, pp_true, 0.0, 20.0)

        # Fit exponential intensity with MLE
        # Initial params: [log(a), b]
        pp_est = fit(
            InhomogeneousPoissonProcess{
                ExponentialIntensity{Float64},Normal,IntegrationConfig
            },
            h,
            [log(2.0), 0.05],
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa ExponentialIntensity
        @test pp_est.mark_dist isa Normal

        # Parameters should be in the right ballpark
        # Allow wide tolerance since this is stochastic data
        @test 0.5 * intensity_true.a <=
            pp_est.intensity_function.a <=
            3.0 * intensity_true.a
        @test abs(pp_est.intensity_function.b - intensity_true.b) < 0.1
    end

    @testset "Fitting - Sinusoidal (MLE)" begin
        # Generate data from sinusoidal process
        intensity_true = SinusoidalIntensity(5.0, 2.0, 2π, 0.0)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Uniform())

        h = simulate(rng, pp_true, 0.0, 10.0)

        # Fit sinusoidal intensity with MLE
        # Initial params: [log(a), b_unconstrained, φ] where a = exp(p1), b = tanh(p2)*a
        pp_est = fit(
            InhomogeneousPoissonProcess{
                SinusoidalIntensity{Float64},Uniform,IntegrationConfig
            },
            h,
            [log(5.0), 0.5, 0.0];
            ω=2π,
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa SinusoidalIntensity
        @test pp_est.mark_dist isa Uniform
        @test pp_est.intensity_function.ω ≈ 2π

        # Check constraint a >= |b| is satisfied
        @test pp_est.intensity_function.a >= abs(pp_est.intensity_function.b)

        # Parameters should be in a reasonable range
        @test 2.0 <= pp_est.intensity_function.a <= 10.0
        @test abs(pp_est.intensity_function.b) <= pp_est.intensity_function.a
    end

    @testset "Log-density and gradients" begin
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        h = simulate(rng, pp, 0.0, 10.0)
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
        intensity_log = PolynomialIntensity([1.0, 0.5]; link=:log)
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @test string(intensity_linear) == "PolynomialIntensity([1.0, 0.5])"
        @test string(intensity_log) == "PolynomialIntensity([1.0, 0.5], link=:log)"
        @test occursin("InhomogeneousPoissonProcess", string(pp))
    end

    @testset "Intensity type constructors and show methods" begin
        @testset "ExponentialIntensity type promotion" begin
            # Test mixed types get promoted
            intensity_mixed = ExponentialIntensity(2, 0.1)
            @test intensity_mixed isa ExponentialIntensity{Float64}
            @test intensity_mixed.a == 2.0
            @test intensity_mixed.b == 0.1

            # Test Float32 types
            intensity_f32 = ExponentialIntensity(2.0f0, 0.1f0)
            @test intensity_f32 isa ExponentialIntensity{Float32}

            # Test show method
            intensity_exp = ExponentialIntensity(2.5, 0.15)
            @test string(intensity_exp) == "ExponentialIntensity(2.5, 0.15)"
        end

        @testset "SinusoidalIntensity show method" begin
            intensity_sin = SinusoidalIntensity(5.0, 2.0, 2π, 0.5)
            expected_str = "SinusoidalIntensity(5.0, 2.0, $(2π), 0.5)"
            @test string(intensity_sin) == expected_str
        end

        @testset "PiecewiseConstantIntensity show method" begin
            breakpoints = [0.0, 1.0, 2.0]
            rates = [1.5, 3.0]
            intensity_pw = PiecewiseConstantIntensity(breakpoints, rates)
            expected_str = "PiecewiseConstantIntensity([0.0, 1.0, 2.0], [1.5, 3.0])"
            @test string(intensity_pw) == expected_str
        end

        @testset "LinearCovariateIntensity show method" begin
            cov1 = t -> t
            cov2 = t -> sin(t)
            intensity_cov = LinearCovariateIntensity(1.0, [0.5, 0.3], [cov1, cov2])
            expected_str = "LinearCovariateIntensity(1.0, [0.5, 0.3], 2 covariates)"
            @test string(intensity_cov) == expected_str

            # Test with more covariates
            cov3 = t -> cos(t)
            intensity_cov3 = LinearCovariateIntensity(
                2.0, [0.1, 0.2, 0.3], [cov1, cov2, cov3]
            )
            expected_str3 = "LinearCovariateIntensity(2.0, [0.1, 0.2, 0.3], 3 covariates)"
            @test string(intensity_cov3) == expected_str3
        end
    end

    @testset "Integrated intensity edge cases" begin
        config = IntegrationConfig()

        @testset "ExponentialIntensity with b ≈ 0" begin
            # When b ≈ 0, should behave like constant function
            intensity_const = ExponentialIntensity(3.0, 1e-12)
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_const, Normal())

            # ∫ 3.0 dt from 0 to 10 = 30.0
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 10.0)
            @test integral ≈ 30.0 rtol = 1e-6
        end

        @testset "ExponentialIntensity with negative b" begin
            # Decreasing exponential: λ(t) = 5*exp(-0.2*t)
            intensity_dec = ExponentialIntensity(5.0, -0.2)
            h_empty = History(Float64[], 0.0, 5.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_dec, Normal())

            # ∫₀⁵ 5*exp(-0.2*t) dt = (5/-0.2)*(exp(-1) - exp(0)) = -25*(exp(-1) - 1)
            expected = (5.0 / -0.2) * (exp(-0.2 * 5.0) - exp(0.0))
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 5.0)
            @test integral ≈ expected rtol = 1e-6
        end

        @testset "PiecewiseConstantIntensity - single region" begin
            # Test when entire integration interval is within a single constant region
            breakpoints = [0.0, 5.0, 10.0, 15.0]
            rates = [2.0, 4.0, 3.0]
            intensity_pw = PiecewiseConstantIntensity(breakpoints, rates)
            h_empty = History(Float64[], 0.0, 15.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_pw, Normal())

            # Integrate within first region [0, 5)
            integral1 = integrated_ground_intensity(pp, h_empty, 1.0, 3.0)
            @test integral1 ≈ 2.0 * (3.0 - 1.0) rtol = 1e-6

            # Integrate within second region [5, 10)
            integral2 = integrated_ground_intensity(pp, h_empty, 6.0, 8.0)
            @test integral2 ≈ 4.0 * (8.0 - 6.0) rtol = 1e-6

            # Integrate within third region [10, 15)
            integral3 = integrated_ground_intensity(pp, h_empty, 11.0, 13.0)
            @test integral3 ≈ 3.0 * (13.0 - 11.0) rtol = 1e-6
        end

        @testset "PiecewiseConstantIntensity - multiple regions" begin
            breakpoints = [0.0, 2.0, 5.0, 8.0]
            rates = [1.0, 3.0, 2.0]
            intensity_pw = PiecewiseConstantIntensity(breakpoints, rates)
            h_empty = History(Float64[], 0.0, 8.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_pw, Normal())

            # Integrate across all regions
            # ∫₀⁸ = 1.0*(2-0) + 3.0*(5-2) + 2.0*(8-5) = 2 + 9 + 6 = 17
            integral_all = integrated_ground_intensity(pp, h_empty, 0.0, 8.0)
            @test integral_all ≈ 17.0 rtol = 1e-6

            # Integrate across first two regions
            # ∫₀⁵ = 1.0*(2-0) + 3.0*(5-2) = 2 + 9 = 11
            integral_partial = integrated_ground_intensity(pp, h_empty, 0.0, 5.0)
            @test integral_partial ≈ 11.0 rtol = 1e-6

            # Integrate across region boundary
            # ∫₁⁶ = 1.0*(2-1) + 3.0*(5-2) + 2.0*(6-5) = 1 + 9 + 2 = 12
            integral_cross = integrated_ground_intensity(pp, h_empty, 1.0, 6.0)
            @test integral_cross ≈ 12.0 rtol = 1e-6
        end

        @testset "LinearCovariateIntensity numerical integration" begin
            # λ(t) = exp(1.0 + 0.5*t + 0.2*sin(t))
            cov1 = t -> t
            cov2 = t -> sin(t)
            intensity_cov = LinearCovariateIntensity(1.0, [0.5, 0.2], [cov1, cov2])
            h_empty = History(Float64[], 0.0, 5.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_cov, Normal())

            # This should use numerical integration
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 5.0)

            # Verify it's positive and finite
            @test integral > 0
            @test isfinite(integral)

            # Rough sanity check: λ(0) = exp(1.0) ≈ 2.718
            # λ(5) = exp(1.0 + 2.5 + 0.2*sin(5)) ≈ exp(3.5 + something small)
            # Integral should be somewhere in a reasonable range
            @test integral > 5.0  # Minimum if constant at λ(0)
            @test integral < 1000.0  # Reasonable upper bound
        end

        @testset "Generic fallback for custom functions" begin
            # Test custom lambda function
            custom_func = t -> 2.0 + 0.5 * t^2
            h_empty = History(Float64[], 0.0, 4.0, Float64[])
            pp = InhomogeneousPoissonProcess(custom_func, Uniform())

            # ∫₀⁴ (2 + 0.5*t²) dt = [2t + 0.5*t³/3]₀⁴ = 8 + 0.5*64/3 = 8 + 32/3 ≈ 18.667
            expected = 2.0 * 4.0 + 0.5 * (4.0^3) / 3.0
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 4.0)
            @test integral ≈ expected rtol = 1e-3

            # Test another custom function with trigonometry
            custom_trig = t -> 3.0 + sin(2π * t)
            pp_trig = InhomogeneousPoissonProcess(custom_trig, Normal())

            # Over one period, sin integrates to 0, so integral ≈ 3*1 = 3
            integral_trig = integrated_ground_intensity(pp_trig, h_empty, 0.0, 1.0)
            @test integral_trig ≈ 3.0 rtol = 1e-3
        end
    end

    @testset "Intensity bounds edge cases" begin
        @testset "ExponentialIntensity bound with b > 0 (increasing)" begin
            intensity_inc = ExponentialIntensity(2.0, 0.1)
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_inc, Normal())

            B, L = ground_intensity_bound(pp, 0.0, h_empty)
            # Max should be at t + lookahead (t=1.0), so λ(1) = 2*exp(0.1) ≈ 2.21
            # With 5% margin: ≈ 2.32
            @test B >= 2.0 * exp(0.1)
            @test B <= 2.0 * exp(0.1) * 1.1  # Check margin isn't too large
            @test L == 1.0
        end

        @testset "ExponentialIntensity bound with b < 0 (decreasing)" begin
            intensity_dec = ExponentialIntensity(5.0, -0.2)
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_dec, Normal())

            B, L = ground_intensity_bound(pp, 2.0, h_empty)
            # Max should be at t=2, so λ(2) = 5*exp(-0.4)
            # With 5% margin
            expected_max = 5.0 * exp(-0.2 * 2.0) * 1.05
            @test B ≈ expected_max rtol = 1e-6
            @test L == 1.0
        end

        @testset "ExponentialIntensity bound with b ≈ 0 (constant)" begin
            intensity_const = ExponentialIntensity(3.0, 1e-12)
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            pp = InhomogeneousPoissonProcess(intensity_const, Normal())

            B, L = ground_intensity_bound(pp, 5.0, h_empty)
            # Should be approximately a * 1.05 = 3.15
            @test B ≈ 3.0 * 1.05 rtol = 1e-6
            @test L == 1.0
        end
    end

    @testset "Marked intensity function" begin
        @testset "Continuous marks" begin
            # Test that intensity(pp, m, t, h) = ground_intensity(pp, t, h) * pdf(mark_dist, m)
            intensity_func = PolynomialIntensity([2.0, 0.3])
            mark_dist = Normal(0.0, 1.0)
            pp = InhomogeneousPoissonProcess(intensity_func, mark_dist)
            h_empty = History(Float64[], 0.0, 10.0, Float64[])

            t = 5.0
            m = 1.5

            # ground_intensity at t=5: λ(5) = 2.0 + 0.3*5 = 3.5
            expected_ground = 2.0 + 0.3 * 5.0

            # intensity(m, t, h) = λ(t) * pdf(Normal(0,1), 1.5)
            expected_full = expected_ground * pdf(mark_dist, m)

            @test ground_intensity(pp, t, h_empty) ≈ expected_ground
            @test intensity(pp, m, t, h_empty) ≈ expected_full

            # Test with different mark value
            m2 = -0.5
            @test intensity(pp, m2, t, h_empty) ≈ expected_ground * pdf(mark_dist, m2)
        end

        @testset "Discrete marks" begin
            # Test with categorical marks
            intensity_func = ExponentialIntensity(3.0, 0.05)
            mark_dist = Categorical([0.3, 0.5, 0.2])
            pp = InhomogeneousPoissonProcess(intensity_func, mark_dist)
            h_empty = History(Float64[], 0.0, 10.0, Float64[])

            t = 2.0
            # ground_intensity at t=2: λ(2) = 3*exp(0.05*2) = 3*exp(0.1)
            expected_ground = 3.0 * exp(0.05 * 2.0)

            @test ground_intensity(pp, t, h_empty) ≈ expected_ground

            # For categorical marks, pdf gives the probability mass
            @test intensity(pp, 1, t, h_empty) ≈ expected_ground * 0.3
            @test intensity(pp, 2, t, h_empty) ≈ expected_ground * 0.5
            @test intensity(pp, 3, t, h_empty) ≈ expected_ground * 0.2
        end
    end

    @testset "Convenience constructors" begin
        @testset "InhomogeneousPoissonProcess without marks" begin
            # Test convenience constructor for unmarked process
            intensity = PolynomialIntensity([1.0, 0.5])
            pp = InhomogeneousPoissonProcess(intensity)

            @test pp isa InhomogeneousPoissonProcess
            @test pp.intensity_function === intensity
            @test pp.mark_dist isa Dirac{Nothing}
            @test pp.mark_dist.value === nothing
            @test pp.integration_config isa IntegrationConfig

            # Test that it simulates correctly
            h = simulate(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test all(m === nothing for m in event_marks(h))
        end

        @testset "InhomogeneousPoissonProcess with custom integration config" begin
            intensity = ExponentialIntensity(2.0, 0.1)
            custom_config = IntegrationConfig(abstol=1e-10, reltol=1e-10, maxiters=5000)
            pp = InhomogeneousPoissonProcess(intensity; integration_config=custom_config)

            @test pp isa InhomogeneousPoissonProcess
            @test pp.intensity_function === intensity
            @test pp.mark_dist isa Dirac{Nothing}
            @test pp.integration_config === custom_config
            @test pp.integration_config.abstol == 1e-10
            @test pp.integration_config.reltol == 1e-10
            @test pp.integration_config.maxiters == 5000
        end
    end

    @testset "IntegrationConfig show method" begin
        @testset "Default config" begin
            config = IntegrationConfig()
            config_str = string(config)
            @test occursin("IntegrationConfig", config_str)
            @test occursin("QuadGKJL", config_str)
            @test occursin("abstol=1.0e-8", config_str)
            @test occursin("reltol=1.0e-8", config_str)
            @test occursin("maxiters=1000", config_str)
        end

        @testset "Custom config" begin
            config = IntegrationConfig(abstol=1e-12, reltol=1e-10, maxiters=10000)
            config_str = string(config)
            @test occursin("IntegrationConfig", config_str)
            @test occursin("abstol=1.0e-12", config_str)
            @test occursin("reltol=1.0e-10", config_str)
            @test occursin("maxiters=10000", config_str)
        end
    end
end
