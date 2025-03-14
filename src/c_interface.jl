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
Option{T} = Union{T, Nothing}
mutable struct State
    model::Option{Lux.StatefulLuxLayer}
    atoms::Option{AnnotedKDTree{Sphere{Float32}, :center}}
end
global_state = State(nothing, nothing)

mutable struct CSphere
    x::Cfloat
    y::Cfloat
    z::Cfloat
    r::Cfloat
end
mutable struct CPoint
    x::Cfloat
    y::Cfloat
    z::Cfloat
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
        path =Base.unsafe_string(path)
        if ispath(path)
            @debug "deserializing"
            data = deserialize(path)
            @debug "deserialized"
            if typeof(data) <: SerializedModel
                global_state.model = production_instantiate(data)
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
            Iterators.map(1:length) do n
                (;x,y,z,r) = unsafe_load(start,n)
                Sphere(Point3f(x, y, z), r)
            end |> StructVector,
            static(:center)
        )
        0
    catch err
        @error "error" err
        2
    end
end

"""
    evaluate_model
"""
Base.@ccallable function eval_model(pt::Ptr{CPoint}, length::Cint)::Float32
    points = map(1:length) do n
        (;x,y,z) = unsafe_load(pt,n)
        Point3f(x, y, z)
    end |> Batch{Vector{Point3f}}
    global_state.model((points, global_state.atoms)) |> cpu_device() |> first
end
