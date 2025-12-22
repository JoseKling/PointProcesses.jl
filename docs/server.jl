using LiveServer

#=
Use this file to serve the documentation locally for development!

Just run: 
    julia --project=docs docs/server.jl

Then open your browser to http://localhost:8000
=#

# Set the directory to "build"; assumes server.jl and the build directory are in the same folder
serve(; dir=joinpath(@__DIR__, "build"))
