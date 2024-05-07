using NearestNeighbors

"""
	state

The global state manipulated by the c interface.
To use, you must first load the weights using `load_weights` and the input atoms using `load_atoms`.
Then you can call eval_model to get the field on a certain point.
"""
mutable struct state
	weights::NamedTuple
	atoms::KDTree
end

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
function load_weights(path::String)::Int
	
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
function load_atoms(start::Ptr{CSphere},length::UInt64)::Int
	
end

"""
    eval_model(x::Float32,y::Float32,z::Float32)::Float32

evaluate the model at coordinates `x` `y` `z`.
"""
function eval_model(x::Float32,y::Float32,z::Float32)::Float32
	
end

@cfunction load_weights Int (String,)
@cfunction load_atoms Int (Ptr{CSphere},UInt32)
@cfunction eval_model Float32 (Float32,Float32,Float32)

