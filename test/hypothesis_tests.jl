h1 = History([1.0, 2.0, 3.0, 4.0], 0.0, 5.0)
h_empty = History(Float64[], 0.0, 2.0)
PP = PoissonProcess{Float64,NoMarks}
pp = PoissonProcess()

@testset "Statistics" begin
    @test statistic(KSDistance{Uniform}, pp, h1) ≈ 0.2
    @test statistic(KSDistance{Exponential}, pp, h1) ≈ 1 - exp(-1)
    @test statistic(KSDistance{Uniform}, pp, h_empty) ≈ 1
    @test statistic(KSDistance{Exponential}, pp, h_empty) ≈ 1
end

h2 = History(collect(0.0:999.0), 0.0, 1000.0)

@testset "BootstrapTest" begin
    @test_throws ArgumentError BootstrapTest(KSDistance{Uniform}, PP, h_empty)

    bt1 = BootstrapTest(KSDistance{Uniform}, PP, h2; n_sims=10000)

    @test bt1.n_sims == 10000
    @test isapprox(bt1.stat, 1/1000, rtol=0.01);
    @test pvalue(bt1) isa Float64
    @test pvalue(bt1) > 0.99
    @test string(bt1) == "BootstrapTest - pvalue = 1.0"

    bt2 = BootstrapTest(KSDistance{Exponential}, PP, h2)

    @test bt2.n_sims == 1000
    @test bt2.stat ≈ 1 - exp(-1);
    @test pvalue(bt2) isa Float64
    @test pvalue(bt2) < 0.01

    test1 = BootstrapTest(KSDistance{Uniform}, PP, h2; n_sims=1000, rng=Random.seed!(1))
    test2 = BootstrapTest(KSDistance{Uniform}, PP, h2; n_sims=1000, rng=Random.seed!(1))

    @test all(sort(test1.sim_stats) .== sort(test2.sim_stats)) # Test reproducibility
end

@testset "MonteCarloTest" begin
    @test_throws ArgumentError MonteCarloTest(KSDistance{Uniform}, PP, h_empty)

    nbt1 = MonteCarloTest(KSDistance{Uniform}, PP, h2; n_sims=1000)
    nbt2 = MonteCarloTest(KSDistance{Uniform}, pp, h2; n_sims=10000)

    @test nbt1.n_sims == 1000
    @test nbt2.n_sims == 10000
    @test isapprox(nbt1.stat, 1/1000, rtol=0.01);
    @test isapprox(nbt1.stat, nbt2.stat, rtol=0.01);
    @test pvalue(nbt1) isa Float64
    @test pvalue(nbt1) > 0.99
    @test pvalue(nbt2) > 0.99
    @test string(nbt1) == "MonteCarloTest - pvalue = 1.0"

    nbt3 = MonteCarloTest(KSDistance{Exponential}, PP, h2)
    nbt4 = MonteCarloTest(KSDistance{Exponential}, pp, h2)

    @test nbt3.n_sims == 1000
    @test nbt4.n_sims == 1000
    @test nbt3.stat ≈ 1 - exp(-1);
    @test nbt3.stat ≈ nbt4.stat;
    @test pvalue(nbt3) isa Float64
    @test pvalue(nbt3) < 0.01
    @test pvalue(nbt4) < 0.01

    test1 = MonteCarloTest(KSDistance{Uniform}, PP, h2; n_sims=1000, rng=Random.seed!(1))
    test2 = MonteCarloTest(KSDistance{Uniform}, PP, h2; n_sims=1000, rng=Random.seed!(1))

    @test all(sort(test1.sim_stats) .== sort(test2.sim_stats)) # Test reproducibility
end

@testset "Inhomogeneous Poisson goodness-of-fit" begin
    # KSDistance with InhomogeneousPoissonProcess relies on time_change.
    # These tests pin down that the integration-based time_change feeds correctly
    # through statistic / MonteCarloTest.

    rng = Random.seed!(31415)
    intensity_true = PolynomialIntensity([2.0, 0.3])
    pp_true = InhomogeneousPoissonProcess(intensity_true)
    h = simulate(rng, pp_true, 0.0, 50.0)

    @testset "Statistic computation" begin
        s_unif = statistic(KSDistance{Uniform}, pp_true, h)
        s_exp = statistic(KSDistance{Exponential}, pp_true, h)
        @test 0.0 <= s_unif <= 1.0
        @test 0.0 <= s_exp <= 1.0
    end

    @testset "MonteCarloTest correct vs. misspecified" begin
        # correct model should fit the data substantially better than a wildly
        # misspecified one.
        mc_true = MonteCarloTest(
            KSDistance{Exponential}, pp_true, h; n_sims=200, rng=Random.seed!(7)
        )
        @test mc_true.n_sims == 200

        pp_wrong = InhomogeneousPoissonProcess(PolynomialIntensity([20.0]))  # constant 20.0
        mc_wrong = MonteCarloTest(
            KSDistance{Exponential}, pp_wrong, h; n_sims=200, rng=Random.seed!(7)
        )

        # The misspecified model's KS statistic should be substantially
        # larger than the correct one's — this is the most direct measure
        # of relative fit quality and is robust to specific p-value seeds.
        @test mc_wrong.stat > 2 * mc_true.stat
        # Correct model's p-value should also exceed the misspecified one's.
        @test pvalue(mc_true) > pvalue(mc_wrong)
    end
end
