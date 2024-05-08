using Lux

function anakin_model()
    a = 5
    b = 5
    adaptator = ToSimpleChainsAdaptor((static(a * b + 2),))
    chain = Chain(Dense(a * b + 2 => 10,
            elu),
        Dense(10 => 1, elu;
            init_weight = (args...) -> glorot_uniform(args...; gain = 1 / 25_0000)))
    Lux.Chain(preprocessing,
        DeepSet(Chain(Encoding(a, b, 1.5f0), chain)), tanh_fast)
end
