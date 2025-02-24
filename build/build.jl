#!julia 
using Pkg
Pkg.activate(@__DIR__)
using PackageCompiler

target_dir = get(ENV, "OUTDIR", "$(@__DIR__)/lib")

println("Creating library in $target_dir")
PackageCompiler.create_library("$(@__DIR__)/../", target_dir;
    precompile_execution_file = "$(@__DIR__)/generate_precompile.jl",
    lib_name = "MLNanoShaperRunner",
    force = true,
    header_files = ["$(@__DIR__)/../src/MLNanoShaperRunner.h"])
