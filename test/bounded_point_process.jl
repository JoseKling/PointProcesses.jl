rng = Random.seed!(63)

intensities = rand(rng, 10)
bpp = BoundedPointProcess(MultivariatePoissonProcess(intensities), 0.0, 1000.0)
h = rand(rng, bpp)

@test min_time(bpp) == 0.0
@test max_time(bpp) == 1000.0
@test ground_intensity(bpp, 0, h) == sum(intensities)
@test mark_distribution(bpp, 100.0, h) == Categorical(intensities / sum(intensities))
@test mark_distribution(bpp, 0.0) == Categorical(intensities / sum(intensities))
@test intensity(bpp, 1, 0, h) == intensities[1]
@test log_intensity(bpp, 2, 1.0, h) == log(intensities[2])

@test ground_intensity_bound(bpp, 243, h) == (sum(intensities), typemax(Int))
@test ground_intensity_bound(bpp, 243.0, h) == (sum(intensities), Inf)
@test integrated_ground_intensity(bpp, h, 342, 598) == sum(intensities) * (598 - 342)
