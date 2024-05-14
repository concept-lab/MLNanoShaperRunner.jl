using Lux
"""
    anakin()

A model inspired by the anakin paper.
The protein is decomposed in pairs of atom near the interest point.
Each pair of atom is turned into a `Preprocessed` data and is then fed to the `Encoding` layer. 
"""
function anakin(; cutoff_radius=3.0f0)
    a = 5
    b = 5
    chain = Chain(Dense(a * b + 2 => 10,
            elu),
        Dense(10 => 1, elu;
            init_weight=(args...) -> glorot_uniform(args...; gain=1 / 25_0000)))
    Lux.Chain(preprocessing, gpu_device(),
        DeepSet(Chain(Encoding(a, b, cutoff_radius), chain)), tanh_fast)
end

function angular_dense(; cutoff_radius=3.0f0)
    chain = Chain(Dense(5 => 10,elu),
        Dense(10 => 1, elu;
            init_weight=(args...) -> glorot_uniform(args...; gain=1 / 25_0000)))
    Lux.Chain(preprocessing, gpu_device(),
        DeepSet(Chain(x -> symetrise(x; cutoff_radius), trace("feature vector"), chain)), tanh_fast)
end
