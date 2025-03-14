using Lux

function select_and_preprocess((point, atoms); cutoff_radius)
    select_and_preprocess(point, atoms; cutoff_radius)
end

const MyType{T <: Number} = StructVector{Sphere{T}, @NamedTuple{center::Vector{Point{3, T}}, r::Vector{T}}, Int64}
function select_and_preprocess(
        point::Batch,
        atoms::AnnotedKDTree{Sphere{T}};
        cutoff_radius::Number
) where {T}
    # neighboord = Folds.map(point.field) do point
    neighboord =Batch(StructVector{Sphere{T}}[])
    for point in point.field
        push!(neighboord.field,select_neighboord(
            point,
            atoms;
            cutoff_radius
        ))
    end
    preprocessing(point, neighboord; cutoff_radius)
end

@inline select_and_preprocess(
    point::Point3,
    atoms;
    cutoff_radius
) = select_and_preprocess(Batch([point]),atoms;cutoff_radius)

function evaluate_if_atoms_in_neighboord(layer, arg::AbstractArray, ps, st; zero_value)
    if length(arg) == 0
        zero_value
    else
        Lux.apply(layer, arg, ps, st)
    end
end

function smoothing_layer(layer)
    Parallel(
        .*,
        layer,
        Lux.WrappedFunction(scale_factor)
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
        van_der_waals_channel::Bool = false,
        smoothing::Bool = true,
        on_gpu::Bool = true,
        cutoff_radius::Float32 = 3.0f0
)
    main_chain = DeepSet(
        Chain(
        WrappedFunction(on_gpu ? gpu_device() : identity),
        main_chain |> (smoothing ? smoothing_layer : identity)
    ),
    )
    Chain(
        PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        main_chain
        |> (van_der_waals_channel ? add_van_der_waals_channel : identity),
        secondary_chain;
        name
    )
end

"""
    tiny_angular_dense(; categorical=false, van_der_waals_channel=false, kargs...)

	`tiny_angular_dense` is a function that generate a lux model.

"""
function tiny_angular_dense(;
        van_der_waals_channel = false,
        smoothing = true,
        kargs...)
    general_angular_dense(
        Chain(
            Dense(6 => 7, relu),
            Dense(7 => 4,relu)
            ),
        Chain(
            relu6,
            LayerNorm((4+van_der_waals_channel,)),
            Dense(4 + van_der_waals_channel => 6, relu),
            LayerNorm((6,)),
            Dense(6 => 1, sigmoid_fast),
        ),
        ;
        name = "tiny_angular_dense" *
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
        van_der_waals_channel = false,
        smoothing = true,
        kargs...)
    general_angular_dense(
        Chain(Dense(6 => 10, elu), Dense(10 => 50, softplus)),
        Chain(
            # BatchNorm(50 + van_der_waals_channel),
            # Base.Broadcast.BroadcastFunction(sqrt),
            Dense(50 + van_der_waals_channel => 10, elu),
            Dense(10 => 1, sigmoid_fast)
        ),
        ;
        name = "light_angular_dense" *
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
        van_der_waals_channel = false,
        smoothing = true,
        kargs...)
    general_angular_dense(
        Chain(Dense(6 => 15, relu), Dense(15 => 100, relu)),
        Chain(
            # BatchNorm(100 + van_der_waals_channel),
            # Base.Broadcast.BroadcastFunction(sqrt),
            LayerNorm((100+van_der_waals_channel,)),
            Dense(100 + van_der_waals_channel => 15,relu),
            LayerNorm((15,)),
            Dense(15 => 1, sigmoid_fast)
        );
        name = "medium_angular_dense" *
               (van_der_waals_channel ? "_v" : "") *
               (smoothing ? "_s" : ""),
        van_der_waals_channel,
        smoothing,
        kargs...
    )
end
function drop_preprocessing(x::Chain)
    if typeof(x[1]) <: PreprocessingLayer
        Chain(NoOpLayer(), map(i -> x[i], 2:length(x))...)
    else
        x
    end
end

get_preprocessing(x::Chain) =
    if typeof(x[1]) <: PreprocessingLayer
        x[1]
    else
        NoOpLayer()
    end

struct SerializedModel
    model::Partial
    parameters::NamedTuple
    states::NamedTuple
end

function production_instantiate((; model, parameters, states)::SerializedModel)
    Lux.StatefulLuxLayer{true}(model(), parameters, states |> Lux.testmode)
end

function get_cutoff_radius(x::Lux.AbstractLuxLayer)
    get_preprocessing(x).fun.kargs[:cutoff_radius]
end
get_cutoff_radius(x::Lux.StatefulLuxLayer) = get_cutoff_radius(x.model)
