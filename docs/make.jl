using Documenter
using DocumenterCitations
using Literate
using PointProcesses
using Random

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(PointProcesses, :DocTestSetup, :(using PointProcesses); recursive=true)

# Process Literate.jl examples
examples_dir = joinpath(@__DIR__, "examples")
output_dir = joinpath(@__DIR__, "src", "examples")

# Create output directory if it doesn't exist
mkpath(output_dir)

# Find all .jl files in the examples directory
example_files = filter(f -> endswith(f, ".jl"), readdir(examples_dir; join=false))

# Process each example file with Literate
example_pages = []
for example_file in sort(example_files)
    input_path = joinpath(examples_dir, example_file)

    # Generate markdown from Literate.jl
    Literate.markdown(input_path, output_dir; documenter=true, credit=false)

    # Create a page entry (remove .jl extension and add .md)
    page_name = replace(example_file, ".jl" => "")
    output_file = "examples/$(page_name).md"

    push!(example_pages, page_name => output_file)
end

# Build pages list dynamically
pages = Any["Home" => "index.md",]

# Add examples section if there are any examples
if !isempty(example_pages)
    push!(pages, "Examples" => example_pages)
end

# Add API reference at the end
push!(pages, "API reference" => "api.md")

makedocs(;
    modules=[PointProcesses],
    authors="Guillaume Dalle, José Kling",
    sitename="PointProcesses.jl",
    format=Documenter.HTML(),
    pages=pages,
    plugins=[bib],
)

deploydocs(; repo="github.com/JoseKling/PointProcesses.jl", devbranch="main")
