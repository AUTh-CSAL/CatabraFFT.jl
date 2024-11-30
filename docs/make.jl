using CatabraFFT
using Documenter

DocMeta.setdocmeta!(CatabraFFT, :DocTestSetup, :(using CatabraFFT); recursive=true)

makedocs(;
    modules=[CatabraFFT],
    authors="Nikos Pitsianis <pitsianis@yahoo.com> and contributors",
    sitename="CatabraFFT.jl",
    format=Documenter.HTML(;
        canonical="https://pitsianis.github.io/CatabraFFT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pitsianis/CatabraFFT.jl",
    devbranch="main",
)
