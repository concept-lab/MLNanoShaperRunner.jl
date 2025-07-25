FROM docker.io/library/julia
ADD https://github.com/concept-lab/MLNanoShaperRunner.jl.git /MLNanoShaperRunner.jl
RUN julia --project=MLNanoShaperRunner.jl -e 'using Pkg; Pkg.instantiate()'
CMD julia --project=MLNanoShaperRunner.jl
