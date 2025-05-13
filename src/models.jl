using Lux

function select_and_preprocess((point, atoms); cutoff_radius)
    select_and_preprocess(point, atoms; cutoff_radius)
end
function select_and_preprocess(max_nb_atoms::Int, (point, atoms); cutoff_radius)
    select_and_preprocess(max_nb_atoms, point, atoms; cutoff_radius)
end

const MyType{T<:Number} = StructVector{Sphere{T},@NamedTuple{center::Vector{Point{3,T}}, r::Vector{T}},Int64}

function select_neighboord_batch(point::Batch, atoms::RegularGrid{T}) where {T}
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
    cutoff_radius::Number
) where {T}
    neighboord = select_neighboord_batch(point, atoms)
    preprocessing(point, neighboord; cutoff_radius)
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
            Dense(6 => 7, relu),
            # Lux.WrappedFunction(trace("pre sum")),
            Dense(7 => 4, relu),
        ),
        Chain(
            # Lux.WrappedFunction(trace("post sum")),
            relu6,
            # NoOpLayer(),
            LayerNorm((4 + van_der_waals_channel,); dims=(1,)),
            Dense(4 + van_der_waals_channel => 6, relu),
            # NoOpLayer(),
            # Lux.WrappedFunction(trace("pre norm")),
            LayerNorm((6,); dims=(1,)),
            Dense(6 => 1, sigmoid_fast),
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
            Dense(6 => 7, relu),
            # Lux.WrappedFunction(trace("pre sum 1")),
            Dense(7 => 4, x -> exp( 1f0*relu6(x)) -1f0),
        ),
        Chain(
            # Lux.WrappedFunction(trace("post sum")),
            Lux.WrappedFunction(Base.Broadcast.BroadcastFunction( x ->log(1f0 + x)*1f0)),
            LayerNorm((4 + van_der_waals_channel,); dims=(1,)),
            Dense(4 + van_der_waals_channel => 6, relu),
            LayerNorm((6,); dims=(1,)),
            Dense(6 => 1, sigmoid_fast),
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
        Chain(Dense(6 => 10, relu), Dense(10 => 50, relu)),
        Chain(
            # BatchNorm(50 + van_der_waals_channel),
            # Base.Broadcast.BroadcastFunction(sqrt),
            relu6,
            LayerNorm((50 + van_der_waals_channel,); dims=(1,)),
            Dense(50 + van_der_waals_channel => 10, relu),
            LayerNorm((10,); dims=(1,)),
            Dense(10 => 1, sigmoid_fast)
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
            Dense(6 => 10, relu),
            Dense(10 => 50, x -> exp( relu6(x)) -1f0),
        ),
        Chain(
            Lux.WrappedFunction(Base.Broadcast.BroadcastFunction( x ->log(1f0 + x))),
            # LayerNorm((4 + van_der_waals_channel,); dims=(1,)),
            Dense(50 + van_der_waals_channel => 10, relu),
            # LayerNorm((6,); dims=(1,)),
            Dense(10 => 1, sigmoid_fast),
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
            Dense(15 => 1, sigmoid_fast)
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

function production_instantiate((; model, parameters, states)::SerializedModel; on_gpu::Bool=false)
    device = on_gpu ? gpu_device() : identity
    Lux.StatefulLuxLayer{true}(model(; on_gpu), parameters |> device, states |> device |> Lux.testmode)
end

function get_cutoff_radius(x::Lux.AbstractLuxLayer)
    get_preprocessing(x).fun.kargs[:cutoff_radius]
end
get_cutoff_radius(x::Lux.StatefulLuxLayer) = get_cutoff_radius(x.model)
