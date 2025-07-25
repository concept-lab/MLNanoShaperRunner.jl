using Lux
activation(x)= max(zero(x), min(one(x),x))
function select_and_preprocess((point, atoms); cutoff_radius)
    select_and_preprocess(point, atoms; cutoff_radius)
end
function select_and_preprocess(max_nb_atoms::Int, (point, atoms); cutoff_radius)
    select_and_preprocess(max_nb_atoms, point, atoms; cutoff_radius)
end

const MyType{T<:Number} = StructVector{Sphere{T},@NamedTuple{center::Vector{Point{3,T}}, r::Vector{T}},Int64}

function select_neighboord_batch(point::Batch, atoms::RegularGrid{T}) where {T}
    return _inrange(StructArray{Sphere{T}},atoms,point) |> Batch
    neighboord = Batch(StructVector{Sphere{T}}[])
    for point in point.field
        push!(neighboord.field, select_neighboord(
            point,
            atoms
        ))
    end
    neighboord
end
function select_and_preprocess(
    max_nb_atoms::Int,
    point::Batch,
    atoms::RegularGrid{T};
    cutoff_radius::Number
) where {T}
    # @assert length(point.field) >= 1
    neighboord = select_neighboord_batch(point, atoms)
    preprocessing(point, neighboord, max_nb_atoms; cutoff_radius)
end

function select_and_preprocess(
    point::Batch,
    atoms::RegularGrid{T};
    cutoff_radius::Number,
    device = identity
) where {T}
    neighboord = select_neighboord_batch(point, atoms)
    preprocessing(device(point), neighboord; cutoff_radius)
end

@inline select_and_preprocess(
    point::Point3,
    atoms;
    cutoff_radius
) = select_and_preprocess(Batch([point]), atoms; cutoff_radius)


function smoothing_layer(layer, smoothing::Bool)
    Parallel(
        .*,
        layer,
        Lux.WrappedFunction(smoothing ? scale_factor : _ -> 1)
    )
end

function add_van_der_waals_channel(main_chain)
    Parallel(
        vcat,
        main_chain,
        WrappedFunction((x -> Float32.(x)) ∘ is_in_van_der_waals)
    )
end
function general_angular_dense(
    main_chain,
    secondary_chain;
    name::String,
    van_der_waals_channel::Bool=false,
    smoothing::Bool=true,
    on_gpu::Bool=true,
    max_nb_atoms::Union{Nothing,Int}=nothing,
    cutoff_radius::Float32=3.0f0
)
    preprocessing_layer = PreprocessingLayer(isnothing(max_nb_atoms) ? Partial(select_and_preprocess; cutoff_radius) : Partial(select_and_preprocess, max_nb_atoms; cutoff_radius))
    main_chain = Chain(
        WrappedFunction(on_gpu ? gpu_device() : identity),
        smoothing_layer(main_chain, smoothing)
    )
    main_chain = isnothing(max_nb_atoms) ? DeepSet(main_chain) : FixedSizeDeepSet(main_chain, max_nb_atoms * (max_nb_atoms + 1) ÷ 2)
    Chain(
        preprocessing_layer,
        main_chain |> (van_der_waals_channel ? add_van_der_waals_channel : identity),
        secondary_chain;
        name,
    )
end

"""
    tiny_angular_dense(; categorical=false, van_der_waals_channel=false, kargs...)

	`tiny_angular_dense` is a function that generate a lux model.

"""
function tiny_angular_dense(;
    van_der_waals_channel=false,
    smoothing=true,
    kargs...)
    general_angular_dense(
        Chain(
            # Lux.WrappedFunction(trace("input")),
            Dense(6 => 8, relu),
            # Lux.WrappedFunction(trace("pre sum")),
            Dense(8 => 16, relu),
        ),
        Chain(
            # Lux.WrappedFunction(trace("post sum")),
            relu6,
            # NoOpLayer(),
            # LayerNorm((16 + van_der_waals_channel,); dims=(1,)),
            Dense(16 + van_der_waals_channel => 32, relu),
            # NoOpLayer(),
            # Lux.WrappedFunction(trace("pre norm")),
            # LayerNorm((8,); dims=(1,)),
            Dense(32 => 1, activation),
        ),
        ;
        name="tiny_angular_dense" *
             (van_der_waals_channel ? "_v" : "") *
             (smoothing ? "_s" : ""),
        van_der_waals_channel,
        smoothing,
        kargs...
    )
end

"""
    tiny_angular_dense(; categorical=false, van_der_waals_channel=false, kargs...)

	`tiny_angular_dense` is a function that generate a lux model.

"""
function tiny_soft_max_angular_dense(;
    van_der_waals_channel=false,
    smoothing=true,
    kargs...)
    general_angular_dense(
        Chain(
            # Lux.WrappedFunction(trace("input")),
            Dense(6 => 8, relu),
            # Lux.WrappedFunction(trace("pre sum 1")),
            Dense(8 => 16, x -> exp( 1f0*relu6(x)) -1f0),
        ),
        Chain(
            # Lux.WrappedFunction(trace("post sum")),
            Lux.WrappedFunction(Base.Broadcast.BroadcastFunction( x ->log(1f0 + x)*1f0)),
            # LayerNorm((16 + van_der_waals_channel,); dims=(1,)),
            Dense(16 + van_der_waals_channel => 32, relu),
            # LayerNorm((6,); dims=(1,)),
            Dense(32 => 1, activation),
        ),
        ;
        name="tiny_soft_max_angular_dense" *
             (van_der_waals_channel ? "_v" : "") *
             (smoothing ? "_s" : ""),
        van_der_waals_channel,
        smoothing,
        kargs...
    )
end

"""
    light_angular_dense(; categorical=false, van_der_waals_channel=false, kargs...)

	`light_angular_dense` is a function that generate a lux model.

"""
function light_angular_dense(;
    van_der_waals_channel=false,
    smoothing=true,
    kargs...)
    general_angular_dense(
        Chain(Dense(6 => 16, relu), Dense(16=> 32, relu)),
        Chain(
            relu6,
            # LayerNorm((50 + van_der_waals_channel,); dims=(1,)),
            Dense(32+ van_der_waals_channel => 64, relu),
            # LayerNorm((10,); dims=(1,)),
            Dense(64=> 1, activation)
        ),
        ;
        name="light_angular_dense" *
             (van_der_waals_channel ? "_v" : "") *
             (smoothing ? "_s" : ""),
        van_der_waals_channel,
        smoothing,
        kargs...
    )
end


"""
    light_soft_max_angular_dense(; categorical=false, van_der_waals_channel=false, kargs...)

	`light_soft_max_angular_dense` is a function that generate a lux model.

"""
function light_soft_max_angular_dense(;
    van_der_waals_channel=false,
    smoothing=true,
    kargs...)
    general_angular_dense(
        Chain(
            Dense(6 => 16, relu),
            Dense(16=> 32, x -> exp(relu6(x)) -1f0),
        ),
        Chain(
            Lux.WrappedFunction(Base.Broadcast.BroadcastFunction( x ->log(1f0 + x))),
            # LayerNorm((4 + van_der_waals_channel,); dims=(1,)),
            Dense(32+ van_der_waals_channel => 64, relu),
            # LayerNorm((6,); dims=(1,)),
            Dense(64=> 1, activation),
        ),
        ;
        name="light_soft_max_angular_dense" *
             (van_der_waals_channel ? "_v" : "") *
             (smoothing ? "_s" : ""),
        van_der_waals_channel,
        smoothing,
        kargs...
    )
end

"""
    medium_angular_dense(; categorical=false, van_der_waals_channel=false, kargs...)

	`medium_angular_dense` is a function that generate a lux model.

"""
function medium_angular_dense(;
    van_der_waals_channel=false,
    smoothing=true,
    kargs...)
    general_angular_dense(
        Chain(Dense(6 => 15, relu), Dense(15 => 100, relu)),
        Chain(
            # BatchNorm(100 + van_der_waals_channel),
            # Base.Broadcast.BroadcastFunction(sqrt),
            relu6,
            LayerNorm((100 + van_der_waals_channel,); dims=(1,)),
            Dense(100 + van_der_waals_channel => 15, relu),
            LayerNorm((15,); dims=(1,)),
            Dense(15 => 1, activation)
        );
        name="medium_angular_dense" *
             (van_der_waals_channel ? "_v" : "") *
             (smoothing ? "_s" : ""),
        van_der_waals_channel,
        smoothing,
        kargs...
    )
end

function drop_preprocessing(x::Chain)::Lux.Chain
    if typeof(x[1]) <: PreprocessingLayer
        Chain(NoOpLayer(), map(i -> x[i], 2:length(x))...)
    else
        x
    end
end

function get_preprocessing(x::Chain)::Lux.AbstractLuxLayer
    if typeof(x[1]) <: PreprocessingLayer
        x[1]
    else
        NoOpLayer()
    end
end


function get_last_chain(x::Chain)::Lux.Chain
    if typeof(x[2]) <: DeepSet
        Chain(NoOpLayer(), NoOpLayer(), map(i -> x[i], 3:length(x))...)
    else
        x
    end
end

function get_last_chain_dim(chain::Lux.Chain)
    c = chain[2].prepross[2].layers[1]
    # @info c
    c[length(c)].out_dims
end

struct SerializedModel
    model::Partial
    parameters::NamedTuple
    states::NamedTuple
end

"""
    production_instantiate(model::SerializedModel;[on_gpu])::Lux.StatefullLuxLayer{true}
Turn a SerializedModel into a Stateful Lux Layer. 
"""
function production_instantiate((; model, parameters, states)::SerializedModel; on_gpu::Bool=false)#::Lux.StatefulLuxLayer{true}
    device = on_gpu ? gpu_device() : identity
    Lux.StatefulLuxLayer{true}(model(; on_gpu), parameters |> device, states |> device |> Lux.testmode)
end

"""
    get_cutoff_radius(x::Lux.AbstractLuxLayer)
extract the cutoff radius of a Lux Model.
"""
function get_cutoff_radius(x::Lux.AbstractLuxLayer)
    get_preprocessing(x).fun.kargs[:cutoff_radius]
end
get_cutoff_radius(x::Lux.StatefulLuxLayer) = get_cutoff_radius(x.model)
