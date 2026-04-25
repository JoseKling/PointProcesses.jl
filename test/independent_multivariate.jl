rng = Random.seed!(63)

pp1 = PoissonProcess(1.0, Normal())
pp2 = InhomogeneousPoissonProcess(PolynomialIntensity([1.0, 0.5]), Exponential())
imp = IndependentMultivariateProcess([pp1, pp2])

@testset "Constructors" begin
    @test imp isa IndependentMultivariateProcess
end

@testset "Interface" begin
    h = History([[0.1, 0.7], [0.3]], 0, 1, [[0.0, 0.1], [0.2]])

    @test contains(string(imp), "IndependentMultivariateProcess")
    @test DensityKind(imp) == HasDensity()

    @test ndims(imp) == 2

    @test ground_intensity(imp, 0, h) ==
        [ground_intensity(pp1, 0, h), ground_intensity(pp2, 0, h)]
    @test ground_intensity(imp, 0, h, 1) == 1.0
    @test mark_distribution(imp, 0, h) == [pp1.mark_dist, pp2.mark_dist]
    @test mark_distribution(imp, 0, h, 2) == pp2.mark_dist
    @test intensity(imp, 0, 0, h) == [intensity(pp1, 0, 0, h), intensity(pp2, 0, 0, h)]
    @test intensity(imp, 0, 0, h, 1) == intensity(pp1, 0, 0, h)
    @test log_intensity(imp, 0, 0, h) ≈
        [log_intensity(pp1, 0, 0, h), log_intensity(pp2, 0, 0, h)]
    @test log_intensity(imp, 0, 0, h, 1) ≈ log_intensity(pp1, 0, 0, h)
    @test ground_intensity_bound(imp, 0.0, h) ==
        [ground_intensity_bound(pp1, 0.0, h), ground_intensity_bound(pp2, 0.0, h)]
    @test ground_intensity_bound(imp, 0.0, h, 1) == ground_intensity_bound(pp1, 0.0, h)
    @test integrated_ground_intensity(imp, h, 0, 1) == [
        integrated_ground_intensity(pp1, h, 0, 1), integrated_ground_intensity(pp2, h, 0, 1)
    ]
    @test integrated_ground_intensity(imp, h, 0, 1, 1) ==
        integrated_ground_intensity(pp1, h, 0, 1)
end

@testset "Simulation" begin
    bpp = BoundedPointProcess(imp, 0.0, 1000.0)
    h1 = simulate(rng, imp, 0.0, 1000.0)
    h2 = simulate(rng, bpp)

    @test nb_events(h1) > 0
    @test nb_events(h2) > 0
    @test ndims(h1) == 2
    @test ndims(h2) == 2
    @test issorted(event_times(h1))
    @test issorted(event_times(h2))

    @test event_marks(h2) isa Vector{<:Real}
    @test event_dims(h2) isa Vector{<:Int}
end
