# Instantiation
@test_throws DomainError HomogeneousPoissonProcess(-1)

pp = HomogeneousPoissonProcess(1)

@test string(pp) == "HomogeneousPoissonProcess{Int64}(1)"

# Simulation
@test simulate(pp, 0, 10) isa History{Float64,Nothing}
@test simulate(Random.seed!(1), pp, BigFloat(0), 10) isa History{BigFloat,Nothing}

# Fit
h = History(rand(10), 0, 1)
pp_fit = fit(HomogeneousPoissonProcess, h)

@test pp_fit.λ ≈ 10
@test pp_fit isa HomogeneousPoissonProcess{Float64}
@test fit(HomogeneousPoissonProcess{BigFloat}, h) isa HomogeneousPoissonProcess{BigFloat}

# Time change
@test time_change(pp, h) == h
@test all(time_change(pp_fit, h).times .≈ h.times .* 10)

# Intensity
@test ground_intensity(pp_fit) == 10
@test intensity(pp_fit) == 10
