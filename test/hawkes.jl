# Constructor
@test HawkesProcess(1, 1, 1) isa HawkesProcess{Int}
@test HawkesProcess(1, 1, 1.0) isa HawkesProcess{Float64}
@test_throws DomainError HawkesProcess(-1, 1, 1)

hp = HawkesProcess(1, 1, 2)
h = History([1.0, 2.0, 4.0], ["a", "b", "c"], 0.0, 5.0)

# Time change
h_transf = time_change(hp, h)
integral =
    (hp.μ * duration(h)) +
    (hp.α / hp.ω) * sum([1 - exp(-hp.ω * (h.tmax - ti)) for ti in h.times])
@test isa(h_transf, typeof(h))
@test h_transf.marks == h.marks
@test h_transf.tmin ≈ 0
@test h_transf.tmax ≈ integral
@test h_transf.times ≈ [1, (2 + (1 - exp(-2)) / 2), 4 + (2 - exp(-2 * 3) - exp(-2 * 2)) / 2]
# Ground intensity
@test ground_intensity(hp, h, 1) == 1
@test ground_intensity(hp, h, 2) == 1 + hp.α * exp(-hp.ω * 1)
# Integrated ground intensity
@test integrated_ground_intensity(hp, h, h.tmin, h.tmax) ≈ integral
@test integrated_ground_intensity(hp, h, 2, 3) ≈
    hp.μ +
      ((hp.α / hp.ω) * (exp(-hp.ω) - exp(-hp.ω * 2))) +
      ((hp.α / hp.ω) * (1 - exp(-hp.ω)))
# Simulation
h_sim = rand(hp, 0.0, 10.0)
@test issorted(h_sim.times)
@test isa(h_sim, History{Nothing,Float64})
# Estimation
hp_est = fit(HawkesProcess, h_sim)
@test isa(hp_est, HawkesProcess)
# logdensityof
@test logdensityof(hp, h) ≈
    sum(log.(hp.μ .+ (hp.α .* [0, exp(-hp.ω), exp(-hp.ω * 2) + exp(-hp.ω * 3)]))) -
      integral
