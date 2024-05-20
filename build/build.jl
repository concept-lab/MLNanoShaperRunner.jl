using PackageCompiler

target_dir = get(ENV, "OUTDIR", "$(@__DIR__)/../MyLibCompiled")

println("Creating library in $target_dir")
PackageCompiler.create_library(".", target_dir;lib_name = "MLNanoShaperRunner")
