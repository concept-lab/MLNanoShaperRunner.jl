using Lux

function select_and_preprocess((point, atoms); cutoff_radius)
    select_and_preprocess(point, atoms; cutoff_radius)
end
function select_and_preprocess(point::Batch, atoms::AnnotedKDTree{Sphere{T}}; cutoff_radius) where T
    neighboord = Folds.map(point.field) do point
		select_neighboord(point, atoms; cutoff_radius)::StructVector{Sphere{T}}
	end |> Batch{Vector{<:StructVector{Sphere{T}}}}
    preprocessing((point, neighboord))
end

function evaluate_if_atoms_in_neighboord(layer, arg::AbstractArray, ps, st; zero_value)
    if length(arg) == 0
        zero_value
    else
        Lux.apply(layer, arg, ps, st)
    end
end

function general_angular_dense(main_chain, secondary_chain; name::String,
    van_der_waals_channel=false, on_gpu=true, cutoff_radius::Float32=3.0f0)
    main_chain = DeepSet(Chain(
        symetrise(; cutoff_radius, device=on_gpu ? gpu_device() : identity),
        main_chain
    ))
    function add_van_der_waals_channel(main_chain)
        Parallel(vcat,
            main_chain,
            WrappedFunction((x -> Float32.(x)) ∘ is_in_van_der_waals))
    end
    Chain(PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        main_chain |> (van_der_waals_channel ? add_van_der_waals_channel : identity),
        secondary_chain;
        name)
end

"""
    tiny_angular_dense(; categorical=false, van_der_waals_channel=false, kargs...)

	`tiny_angular_dense` is a function that generate a lux model.

"""
function tiny_angular_dense(; van_der_waals_channel=false, kargs...)
    general_angular_dense(
        Parallel(.*,
            Chain(Dense(6 => 7, elu),
                Dense(7 => 4, elu)),
            Lux.WrappedFunction(scale_factor)
        ),
        Chain(
            BatchNorm(4 + van_der_waals_channel),
            Dense(4 + van_der_waals_channel => 6, elu),
            Dense(6 => 1, sigmoid_fast));
        name="tiny_angular_dense_" *
             (van_der_waals_channel ? "v" : ""),
        van_der_waals_channel, kargs...)
end

function light_angular_dense(; van_der_waals_channel=false, kargs...)
    general_angular_dense(
        Parallel(.*,
            Chain(Dense(6 => 10, elu),
                Dense(10 => 5, elu)),
            Lux.WrappedFunction(scale_factor)
        ),
        Chain(
            BatchNorm(5 + van_der_waals_channel),
            Dense(5 + van_der_waals_channel => 10, elu),
            Dense(10 => 1, sigmoid_fast));
        name="light_angular_dense_" *
             (van_der_waals_channel ? "v" : ""),
        van_der_waals_channel, kargs...)
end

function medium_angular_dense(;
    van_der_waals_channel=false, kargs...)
    general_angular_dense(
        Parallel(.*,
            Chain(Dense(6 => 15, elu),
                Dense(15 => 5, elu)),
            Lux.WrappedFunction(scale_factor)
        ),
        Chain(
            BatchNorm(10 + van_der_waals_channel),
            Dense(10 + van_der_waals_channel => 5; use_bias=false),
            Dense(5 => 10, elu),
            Dense(10 => 1, sigmoid_fast));
        name="medium_angular_dense_" *
             (van_der_waals_channel ? "v" : ""),
        van_der_waals_channel,
        kargs...)
end
drop_preprocessing(x::Chain) =
    if typeof(x[1]) <: PreprocessingLayer
        Chain(NoOpLayer(), x[2:end])
    else
        x
    end

get_preprocessing(x::Chain) =
    if typeof(x[1]) <: PreprocessingLayer
        x[1]
    else
        NoOpLayer()
    end

struct SerializedModel
    model::Partial
    weights::NamedTuple
end
function get_cutoff_radius(x::Lux.AbstractExplicitLayer)
    get_preprocessing(x).fun.kargs[:cutoff_radius]
end
get_cutoff_radius(x::Lux.StatefulLuxLayer) = get_cutoff_radius(x.model)
