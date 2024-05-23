using NearestNeighbors
using Serialization
using Lux
using GeometryBasics
using MLNanoShaperRunner
using StructArrays
using LuxCore

"""
	state

The global state manipulated by the c interface.
To use, you must first load the weights using `load_weights` and the input atoms using `load_atoms`.
Then you can call eval_model to get the field on a certain point.
"""
Option{T} = Union{T,Nothing}
mutable struct State
    weights::Option{NamedTuple}
	model::Option{Lux.AbstractExplicitLayer}
    atoms::Option{AnnotedKDTree{Sphere{Float32},:center}}
    cutoff_radius::Float32
end
global_state = State(nothing, nothing, nothing, 3.0)

mutable struct CSphere
    x::Float32
    y::Float32
    z::Float32
    r::Float32
end
CSphere((;center,r)::Sphere) = CSphere(center...,r)

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
			@debug "deserializing"
            data = deserialize(path)
			@debug "deserialized"
            if typeof(data) <:NamedTuple 
                global_state.weights = data
                0
            else
                @error "wrong type, expected Lux.Experimental.TrainState, got" typeof(data)
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
Base.@ccallable function load_atoms(start::Ptr{CSphere}, length::Int)::Int
    try
        global_state.atoms = AnnotedKDTree(Iterators.map(unsafe_wrap(
            Array, start, length)) do (; x, y, z, r)
            Sphere(Point3f(x, y, z), r)
		end |>StructVector ,static(:center))
		0
    catch err
        @error "error" err
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
		global_state.model = angular_dense(;cutoff_radius)
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
	neighbors = select_neighboord(point,global_state.atoms;cutoff_radius = global_state.cutoff_radius)
    LuxCore.stateless_apply(global_state.model, ModelInput(point, neighbors),
        global_state.weights )
end
