@testset "Distributions.jl" begin
    h = History(rand(10), 0, 1, rand(10))
    md = Normal()
    @test md isa PointProcessMarkDistribution
    @test mark_distribution(md, 0.0, h) == Normal()
    @test fit(Normal, h) isa Normal
    @test fit(Exponential, h) isa Exponential
    @test densityof(md, 0.0, h, 0.0) == densityof(md, 0.0)
    sample1 = sample_mark(Random.seed!(1), md, 0.0, h)
    sample2 = rand(Random.seed!(1), Normal())
    @test sample1 == sample2
end

@testset "NoMarks" begin
    h = History(rand(10), 0, 1, rand(10))
    md = NoMarks()
    @test md isa AbstractMarkDistribution
    @test md isa PointProcessMarkDistribution 
    @test mark_distribution(md, 0.0, h) isa Dirac{Nothing}
    @test fit(NoMarks, h) == NoMarks()
    @test fit(NoMarks, rand(10), rand(10)) == NoMarks()
    @test densityof(md, 0.0, h, 0.0) == 0.0
    @test densityof(md, 0.0, h, nothing) == 1.0
    @test sample_mark(md, 0.0, h) === nothing
    @test eltype(NoMarks()) == Nothing
end