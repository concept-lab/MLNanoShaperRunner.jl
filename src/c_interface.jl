using NearestNeighbors
using Serialization
using Lux
using GeometryBasics
using MLNanoShaperRunner
using StructArrays
using LuxCore
using Random

"""
	state

The global state manipulated by the c interface.
To use, you must first load the weights using `load_weights` and the input atoms using `load_atoms`.
Then you can call eval_model to get the field on a certain point.
"""
Option{T} = Union{T,Nothing}
mutable struct State
    model::Option{Lux.StatefulLuxLayer}
    atoms::Option{AnnotedKDTree{Sphere{Float32},:center}}
    cutoff_radius::Float32
end
global_state = State(nothing, nothing, 3.0)

mutable struct CSphere
    x::Float32
    y::Float32
    z::Float32
    r::Float32
end
CSphere((; center, r)::Sphere) = CSphere(center..., r)

"""
    load_model(path::String)::Cint

Load the model from a `MLNanoShaperRunner.SerializedModel` serialized state at absolute path `path`.

# Return an error status:
- 0: OK
- 1: file not found
- 2: file could not be deserialized properly
- 3: unknow error
"""
function load_model end
"""
	load_atoms(start::Ptr{CSphere},length::Cint)::Cint

Load the atoms into the julia model.
Start is a pointer to the start of the array of `CSphere` and `length` is the length of the array

# Return an error status:
- 0: OK
- 1: data could not be read
- 2: unknow error
"""
function load_atoms end

"""
    eval_model(x::Float32,y::Float32,z::Float32)::Float32

evaluate the model at coordinates `x` `y` `z`.
"""
function eval_model end

Base.@ccallable function load_model(path::Cstring)::Cint
    try
        if ispath(path)
            @debug "deserializing"
            data = deserialize(path)
            @debug "deserialized"
            if typeof(data) <: SerializedModel
                global_state.model = StatefulLuxLayer(data.model(), data.parameters,
                    Lux.initialstates(MersenneTwister(42), data.model()))
                global_state.cutoff_radius = get_cutoff_radius(global_state.model)
                0
            else
                @error "wrong type, expected MLNanoShaperRunner.SerializedModel, got" typeof(data)
                2
            end
        else
            @error "file not found" path
            1
        end
    catch err
        @error err
        3
    end
end
Base.@ccallable function load_atoms(start::Ptr{CSphere}, length::Cint)::Cint
    try
        global_state.atoms = AnnotedKDTree(
            Iterators.map(unsafe_wrap(
                Array, start, length)) do (; x, y, z, r)
                Sphere(Point3f(x, y, z), r)
            end |> StructVector,
            static(:center))
        0
    catch err
        @error "error" err
        2
    end
end

Base.@ccallable function eval_model(x::Float32, y::Float32, z::Float32)::Float32
    point = Point3f(x, y, z)
    if distance(point, global_state.atoms.tree) >= global_state.cutoff_radius
        0.0f0
    else
        only(model((Point3f(x), global_state.atoms))) - 0.5f0
    end
end
