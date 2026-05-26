using Aqua
using DensityInterface
using Documenter
using Distributions
using ForwardDiff
using LinearAlgebra
using Pkg
using PointProcesses
using Random
using Statistics
using StatsAPI
using Test

Random.seed!(63)

DocMeta.setdocmeta!(PointProcesses, :DocTestSetup, :(using PointProcesses); recursive=true)

@testset verbose = true "PointProcesses.jl" begin
    @testset verbose = false "Code quality (Aqua.jl)" begin
        Aqua.test_all(PointProcesses; ambiguities=false, deps_compat=(; check_extras=false))
    end
    @testset verbose = false "Code Linting" begin
        # Skip JET on Julia pre-releases (where JET typically hasn't caught up
        # yet and there is no compatible JET version to resolve against). JET is
        # not listed in test/Project.toml's [deps] for the same reaso, having
        # it there would make `Pkg.test` fail at the resolution step on
        # prereleases, before this guard ever runs. Install on demand only when
        # we're on a stable Julia.
        if isempty(VERSION.prerelease)
            Pkg.add("JET")
            @eval using JET
            JET.test_package(PointProcesses; target_modules=(PointProcesses,))
        end
    end
    @testset verbose = false "Doctests" begin
        doctest(PointProcesses)
    end
    @testset verbose = true "History" begin
        include("history.jl")
    end
    @testset verbose = false "MarkDistributions" begin
        include("mark_distributions.jl")
    end
    @testset verbose = true "Bounded" begin
        include("bounded_point_process.jl")
    end
    @testset verbose = true "IndependentMultivariate" begin
        include("independent_multivariate.jl")
    end
    @testset verbose = true "Poisson" begin
        @testset verbose = true "Homogeneous" begin
            include("poisson_process.jl")
        end
        @testset verbose = true "Inhomogeneous" begin
            include("inhomogeneous_poisson_process.jl")
        end
        @testset verbose = true "Multivariate" begin
            include("multivariate_poisson_process.jl")
        end
    end
    @testset verbose = true "Hawkes" begin
        include("hawkes.jl")
    end
    @testset verbose = true "PPTests" begin
        include("hypothesis_tests.jl")
    end
end
