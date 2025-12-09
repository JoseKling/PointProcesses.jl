h1 = History([1, 2, 3, 4], 0, 5)
h_empty = History(Float64[], 0, 2)
PP = UnivariatePoissonProcess{Float32}
pp = PoissonProcess()

@testset "Statistics" begin
    @test statistic(KSDistance{Uniform}, pp, h1) ≈ 0.2
    @test statistic(KSDistance{Exponential}, pp, h1) ≈ 1 - exp(-1)

    @test statistic(KSDistance{Uniform}, pp, h_empty) ≈ 1
    @test statistic(KSDistance{Exponential}, pp, h_empty) ≈ 1
end

h2 = History(collect(0:999), 0, 1000)

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

@testset "NoBootstrapTest" begin
    @test_throws ArgumentError NoBootstrapTest(KSDistance{Uniform}, PP, h_empty)

    nbt1 = NoBootstrapTest(KSDistance{Uniform}, PP, h2; n_sims=1000)
    nbt2 = NoBootstrapTest(KSDistance{Uniform}, pp, h2; n_sims=10000)

    @test nbt1.n_sims == 1000
    @test nbt2.n_sims == 10000
    @test isapprox(nbt1.stat, 1/1000, rtol=0.01);
    @test isapprox(nbt1.stat, nbt2.stat, rtol=0.01);
    @test pvalue(nbt1) isa Float64
    @test pvalue(nbt1) > 0.99
    @test pvalue(nbt2) > 0.99
    @test string(nbt1) == "NoBootstrapTest - pvalue = 1.0"

    nbt3 = NoBootstrapTest(KSDistance{Exponential}, PP, h2)
    nbt4 = NoBootstrapTest(KSDistance{Exponential}, pp, h2)

    @test nbt3.n_sims == 1000
    @test nbt4.n_sims == 1000
    @test nbt3.stat ≈ 1 - exp(-1);
    @test nbt3.stat ≈ nbt4.stat;
    @test pvalue(nbt3) isa Float64
    @test pvalue(nbt3) < 0.01
    @test pvalue(nbt4) < 0.01

    test1 = NoBootstrapTest(KSDistance{Uniform}, PP, h2; n_sims=1000, rng=Random.seed!(1))
    test2 = NoBootstrapTest(KSDistance{Uniform}, PP, h2; n_sims=1000, rng=Random.seed!(1))

    @test all(sort(test1.sim_stats) .== sort(test2.sim_stats)) # Test reproducibility
end
