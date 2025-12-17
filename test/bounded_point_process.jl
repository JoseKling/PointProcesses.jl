rng = Random.seed!(63)

intensities = rand(rng, 10)
bpp = BoundedPointProcess(PoissonProcess(intensities), 0.0, 1000.0)
h = simulate(rng, bpp)

@test min_time(bpp) == 0.0
@test max_time(bpp) == 1000.0
@test ground_intensity(bpp, h, 0) == sum(intensities)
@test mark_distribution(bpp, h, 100.0) == Categorical(intensities / sum(intensities))
@test mark_distribution(bpp, 0.0) == Categorical(intensities / sum(intensities))
@test intensity(bpp, 1, h, 0) == intensities[1]
@test log_intensity(bpp, 2, h, 1.0) == log(intensities[2])

@test ground_intensity_bound(bpp, 243, h) == (sum(intensities), typemax(Int))
@test ground_intensity_bound(bpp, 243.0, h) == (sum(intensities), Inf)
@test integrated_ground_intensity(bpp, h, 342, 598) == sum(intensities) * (598 - 342)
