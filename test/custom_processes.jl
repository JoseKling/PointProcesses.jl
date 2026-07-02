@testset "Univariate" begin
    struct FakePoisson <: AbstractUnivariateProcess
        λ::Float64
        mark_dist::NoMarks
    end

    PointProcesses.ground_intensity(fp::FakePoisson, t, h::History) = fp.λ

    function PointProcesses.integrated_ground_intensity(fp::FakePoisson, h::History, a, b)
        return fp.λ * (b - a)
    end

    PointProcesses.ground_intensity_bound(fp::FakePoisson, t, h::History) = (fp.λ, Inf)

    fp = FakePoisson(1.0, NoMarks())
    h = simulate(fp, 0.0, 10.0)

    @test h isa History
    @test ndims(h) == 1
    @test ndims(fp) == 1
    @test intensity(fp, nothing, 1.0, h) == 1.0
    @test log_intensity(fp, nothing, 1.0, h) == 0.0
    @test mark_distribution(fp, 1.0, h) == Dirac(nothing)
    @test sample_mark(fp, 1.0, h) === nothing
    @test time_change(h, fp).times == h.times
    @test logdensityof(fp, h) == -10.0

    fp_bounded = BoundedPointProcess(FakePoisson(1.0, NoMarks()), 0.0, 10.0)
    @test simulate(fp_bounded) isa History
    @test time_change(h, fp_bounded).times == h.times
end

@testset "Multivariate" begin
    struct FakeMultivariatePoisson <: AbstractMultivariateProcess
        λ::Vector{Float64}
        mark_dist::Vector{NoMarks}
    end

    PointProcesses.mark_distribution(::FakeMultivariatePoisson, t, ::History, ::Int) =
        Dirac(nothing)

    PointProcesses.ground_intensity(fmp::FakeMultivariatePoisson, t, h::History, d::Int) =
        fmp.λ[d]

    function PointProcesses.integrated_ground_intensity(
        fmp::FakeMultivariatePoisson, h::History, a, b, d::Int
    )
        return fmp.λ[d] * (b - a)
    end

    PointProcesses.ground_intensity_bound(
        fmp::FakeMultivariatePoisson, t, h::History, d::Int
    ) = (fmp.λ[d], Inf)

    fmp = FakeMultivariatePoisson([1.0, 2.0], [NoMarks(), NoMarks()])
    h = History([rand(10), rand(10)], 0.0, 1.0)

    @test h isa History
    @test ndims(h) == 2
    @test ndims(fmp) == 2
    @test mark_distribution(fmp, 1.0, h) == [Dirac(nothing), Dirac(nothing)]
    @test ground_intensity(fmp, 1.0, h) == [1.0, 2.0]
    @test integrated_ground_intensity(fmp, h, 0.0, 2.0) == [2.0, 4.0]
    @test ground_intensity_bound(fmp, 1.0, h) == [(1.0, Inf), (2.0, Inf)]
    @test intensity(fmp, nothing, 1.0, h, 1) == 1.0
    @test intensity(fmp, nothing, 1.0, h) == [1.0, 2.0]
    @test log_intensity(fmp, nothing, 1.0, h, 1) == 0.0
    @test log_intensity(fmp, nothing, 1.0, h) == [0.0, log(2.0)]
    h_transf = time_change(h, fmp)
    @test event_times(h_transf, 1) == event_times(h, 1)
    @test event_times(h_transf, 2) == event_times(h, 2) .* 2
end
