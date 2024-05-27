using Lux

function select_and_preprocess((point, atoms); cutoff_radius)
    atoms = select_neighboord(point, atoms; cutoff_radius)
    preprocessing(ModelInput(point, atoms))
end
"""
    anakin()

A model inspired by the anakin paper.
The protein is decomposed in pairs of atom near the interest point.
Each pair of atom is turned into a `Preprocessed` data and is then fed to the `Encoding` layer. 
"""
function anakin(; cutoff_radius::Float32 = 3.0f0)
    a = 5
    b = 5
    chain = Chain(Dense(a * b + 2 => 10,
            elu),
        Dense(10 => 1, elu;
            init_weight = Partial(glorot_uniform; gain = 1 / 25_0000)))
    Lux.Chain(
        PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        DeepSet(Chain(Base.Fix2(reshape, (1, 1, :)),
            Encoding(a, b, cutoff_radius), gpu_device(), chain)), tanh_fast; name = "anakin")
end

function angular_dense(; cutoff_radius::Float32 = 3.0f0)
    chain = Chain(Dense(5 => 10, elu),
        Dense(10 => 1, elu;
            init_weight = Partial(glorot_uniform; gain = 1 / 25_0000)))
    Lux.Chain(
        PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        DeepSet(Chain(symetrise(; cutoff_radius), gpu_device(),
            trace("feature vector"), chain)), tanh_fast; name = "angular_dense")
end

function deep_angular_dense(; cutoff_radius::Float32 = 3.0f0)
    chain = Chain(
        Dense(5 => 30; use_bias = false),
        Dense(30 => 10, elu))
    Chain(PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        DeepSet(Chain(symetrise(; cutoff_radius), gpu_device(), chain)),
        Dense(10 => 30; use_bias = false),
        Dense(30 => 10, elu),
        Dense(10 => 1, elu;
            init_weight = Partial(glorot_uniform; gain = 1 / 25_0000)),
        tanh_fast;
        name = "deep_angular_dense")
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
