using Lux

select_and_preprocess((point, atoms); cutoff_radius) =select_and_preprocess(point,atoms;cutoff_radius)
function select_and_preprocess(point::Batch, atoms; cutoff_radius)
	Batch(select_and_preprocess.(point,Ref(atoms);cutoff_radius))
end
function select_and_preprocess(point::Point, atoms; cutoff_radius)
    atoms = select_neighboord(point, atoms; cutoff_radius)
    preprocessing(ModelInput(point, atoms))
end

function evaluate_if_atoms_in_neighboord(layer, arg::AbstractArray, ps, st; zero_value)
    if length(arg) == 0
        zero_value
    else
        Lux.apply(layer, arg, ps, st)
    end
end

"""
    anakin()

A model inspired by the anakin paper.
The protein is decomposed in pairs of atom near the interest point.
Each pair of atom is turned into a `Preprocessed` data and is then fed to the `Encoding` layer. 
"""
function anakin(; cutoff_radius::Float32=3.0f0)
    a = 5
    b = 5
    chain = Chain(Dense(a * b + 2 => 10,
            elu),
        Dense(10 => 1, elu;
            init_weight=Partial(glorot_uniform; gain=1 / 25_0000)))
    Lux.Chain(
        PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        DeepSet(Chain(Base.Fix2(reshape, (1, 1, :)),
            Encoding(a, b, cutoff_radius), gpu_device(), chain)), tanh_fast; name="anakin")
end

function angular_dense(; cutoff_radius::Float32=3.0f0)
    chain = Chain(Dense(5 => 10, elu),
        Dense(10 => 1, elu))
    Lux.Chain(
        PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        DeepSet(Chain(symetrise(; cutoff_radius), gpu_device(),
            chain)), tanh_fast; name="angular_dense")
end

function deep_angular_dense(; cutoff_radius::Float32=3.0f0)
    chain = Chain(
        Dense(5 => 30, elu),
        Dense(30 => 10, elu))
    Chain(PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        DeepSet(Chain(symetrise(; cutoff_radius), gpu_device(), chain)),
        Dense(10 => 30, elu),
        Dense(30 => 10, elu),
        Dense(10 => 1, elu),
        tanh_fast;
        name="deep_angular_dense")
end

function general_angular_dense(main_chain, secondary_chain; name::String,
    van_der_wal_channel=false, on_gpu=true, cutoff_radius::Float32=3.0f0)
    main_chain = DeepSet(Chain(symetrise(; cutoff_radius),
        on_gpu ? gpu_device() : NoOpLayer(), Parallel((.*), main_chain, Lux.WrappedFunction(scale_factor))
    ))
    function add_van_der_wal_channel(main_chain)
        Parallel(vcat,
            main_chain,
            WrappedFunction((x -> Float32.(x)) ∘ is_in_van_der_val))
    end
    Chain(PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        main_chain |> (van_der_wal_channel ? add_van_der_wal_channel : identity),
        secondary_chain;
        name)
end

function tiny_angular_dense(; categorical=false, van_der_wal_channel=false, kargs...)
    general_angular_dense(
        Chain(Dense(6 => 7, elu),
            Dense(7 => 4, elu)),
        Chain(Dense(4 + van_der_wal_channel => 6, elu),
            Dense(6 => 1, categorical ? identity : tanh_fast));
        name="tiny_angular_dense_" * (categorical ? "c" : "") *
             (van_der_wal_channel ? "v" : ""),
        van_der_wal_channel, kargs...)
end

function light_angular_dense(; categorical=false, van_der_wal_channel=false, kargs...)
    general_angular_dense(
        Chain(Dense(6 => 10, elu),
            Dense(10 => 5, elu)),
        Chain(Dense(5 + van_der_wal_channel => 10, elu),
            Dense(10 => 1, categorical ? identity : tanh_fast));
        name="light_angular_dense_" * (categorical ? "c" : "") *
             (van_der_wal_channel ? "v" : ""),
        van_der_wal_channel, kargs...)
end

function medium_angular_dense(; categorical=false, van_der_wal_channel=false, kargs...)
    general_angular_dense(Chain(
            Dense(6 => 15, elu),
            Dense(15 => 10, elu)),
        Chain(
            Dense(10 + van_der_wal_channel => 5; use_bias=false),
            Dense(5 => 10, elu),
            Dense(10 => 1, categorical ? identity : tanh_fast));
        name="medium_angular_dense_" *
             (categorical ? "c" : "") *
             (van_der_wal_channel ? "v" : ""),
        van_der_wal_channel,
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
