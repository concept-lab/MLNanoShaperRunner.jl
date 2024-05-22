#!julia 
using Pkg
Pkg.activate(@__DIR__)
using PackageCompiler

target_dir = get(ENV, "OUTDIR", "$(@__DIR__)/lib")

println("Creating library in $target_dir")
PackageCompiler.create_library("$(@__DIR__)/../", target_dir;lib_name = "MLNanoShaperRunner",force=true)
