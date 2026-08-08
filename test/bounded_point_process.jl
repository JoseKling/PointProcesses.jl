rng = Random.seed!(63)

intensities = rand(rng, 10)
bpp = BoundedPointProcess(PoissonProcess(intensities, Normal()), 0.0, 1000.0)
h = simulate(rng, bpp)

@test min_time(bpp) == 0.0
@test max_time(bpp) == 1000.0
@test ground_intensity(bpp, 0, h) ≈ intensities
@test mark_distribution(bpp, 100.0, h, 1) == Normal()
@test mark_distribution(bpp, 0.0, h) == fill(Normal(), length(intensities))
@test intensity(bpp, 0.0, 0.0, h, 1) ≈ pdf(Normal(), 0.0) * intensities[1]
@test log_intensity(bpp, 1.0, 1.0, h, 2) ≈ log(pdf(Normal(), 1.0)) + log(intensities[2])

@test all(ground_intensity_bound(bpp, 243.0, h, 1) .≈ (intensities[1], Inf))
@test integrated_ground_intensity(bpp, h, 342, 598) ≈ intensities * (598 - 342)
