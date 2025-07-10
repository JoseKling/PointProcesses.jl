hp = HawkesProcess(1, 1, 2)
h = History([1.0, 2.0, 4.0], ["a", "b", "c"], 0.0, 5.0)

# Time change
h_transf = time_change(hp, h)
@test isa(h_transf, typeof(h))
@test h_transf.marks == h.marks
@test h_transf.tmin ≈ 0
@test h_transf.tmax ≈
    (hp.μ * duration(h)) +
      (hp.α / hp.β) * sum([1 - exp(-hp.β * (h.tmax - ti)) for ti in h.times])
@test h_transf.times ≈ [1, (2 + (1 - exp(-2)) / 2), 4 + (2 - exp(-2 * 3) - exp(-2 * 2)) / 2]
# Integrated ground intensity
@test integrated_ground_intensity(hp, h, h.tmin, h.tmax) ≈ h_transf.tmax
@test integrated_ground_intensity(hp, h, 2, 3) ≈
    hp.μ +
      ((hp.α / hp.β) * (exp(-hp.β) - exp(-hp.β * 2))) +
      ((hp.α / hp.β) * (1 - exp(-hp.β)))
# Simulation
h_sim = rand(hp, 0.0, 10.0)
@test issorted(h_sim.times)
@test isa(h_sim, History{Nothing,Float64})
# Estimation
hp_est = fit(HawkesProcess, h_sim)
@test isa(hp_est, HawkesProcess)
