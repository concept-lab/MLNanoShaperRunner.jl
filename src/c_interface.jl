using NearestNeighbors
using Serialization
using Lux
using GeometryBasics
using StructArrays

"""
	state

The global state manipulated by the c interface.
To use, you must first load the weights using `load_weights` and the input atoms using `load_atoms`.
Then you can call eval_model to get the field on a certain point.
"""
Option{T} = Union{T, Nothing}
mutable struct State
    weights::Option{Lux.Experimental.TrainState}
    atoms_tree::Option{KDTree{Point3f}}
    atoms::Option{StructVector{Sphere{Float32}}}
    cutoff_radius::Float32
end
global_state = State(nothing, nothing, nothing, 3.0)

struct CSphere
    x::Float32
    y::Float32
    z::Float32
    r::Float32
end

"""
    load_weights(path::String)::Int

Load the model `parameters` and `model` from a serialised training state at absolute path `path`.

# Return an error status:
- 0: OK
- 1: file not found
- 2: file could not be deserialized properly
- 3: unknow error
"""
function load_weights end
Base.@ccallable function load_weights(path::String)::Int
    try
        if ispath(path)
            data = deserialize(path)
            if typeof(data) <: Lux.Experimental.TrainState
                global_state.weights = deserialize(path)
                0
            else
                @error "file not found" path
                2
            end
        else
            @error "wrong type, expected Lux.Experimental.TrainState, got" typeof(data)
            1
        end
    catch err
        @error err
        3
    end
end

"""
    load_atoms(start::Ptr{CSphere},length::UInt32)::Int

Load the atoms into the julia model.
Start is a pointer to the start of the array of `CSphere` and `length` is the length of the array

# Return an error status:
- 0: OK
- 1: data could not be read
- 2: unknow error
"""
function load_atoms end
Base.@ccallable function load_atoms(start::Ptr{CSphere}, length::UInt64)::Int
    try
        global_state.atoms = Iterators.map(unsafe_wrap(
            Array, start, length)) do (; x, y, z, r)
            Sphere(Point3f(x, y, z), r)
        end |> StridedVector
        global_state.atoms = KDTree(data; reorder = false)
    catch err
        @error err
        2
    end
end

"""
    set_cutoff_radius(cutoff_radius::Float32)::Int

Set the cutoff_radius value for inference.
# Return an error status:
- 0: OK
- 1: formatting error
"""
function set_cutoff_radius end
Base.@ccallable function set_cutoff_radius(cutoff_radius::Float32)::Int
    if cutoff_radius >= 0
        global_state.cutoff_radius = cutoff_radius
        0
    else
        1
    end
end

"""
    eval_model(x::Float32,y::Float32,z::Float32)::Float32

evaluate the model at coordinates `x` `y` `z`.
"""
function eval_model end
Base.@ccallable function eval_model(x::Float32, y::Float32, z::Float32)::Float32
    point = Point3f(x, y, z)
    neighbors = global_state.atoms[inrange(
        global_state.atoms_tree, point, global_state.cutoff_radius)]
    Lux.eval(global_state.weights.model, ModelInput(point, neighbors),
        global_state.weights.parameters, global_state.weights.state)
end
