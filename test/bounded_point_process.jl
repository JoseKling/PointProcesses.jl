rng = Random.seed!(63)

intensities = rand(rng, 10)
bpp = BoundedPointProcess(PoissonProcess(intensities), 0.0, 1000.0)
h = simulate(rng, bpp)

@test min_time(bpp) == 0.0
@test max_time(bpp) == 1000.0
@test ground_intensity(bpp, 0, h) == intensities
@test mark_distribution(bpp, 100.0, h) == fill(Dirac(nothing), 10)
@test mark_distribution(bpp, 0.0, h, 1) == Dirac(nothing)
@test intensity(bpp, 1, 0, h, 1) == 0
@test log_intensity(bpp, 2, 1.0, h, 2) == -Inf

@test ground_intensity_bound(bpp, 243, h) == [(intensity, typemax(Int)) for intensity in intensities]
@test ground_intensity_bound(bpp, 243.0, h) == [(intensity, Inf) for intensity in intensities]
@test integrated_ground_intensity(bpp, h, 342, 598) == intensities .* (598 - 342)
