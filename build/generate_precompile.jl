using MLNanoShaperRunner
using StructArrays

function basic_inference()
    @assert MLNanoShaperRunner.load_weights("$(homedir())/datasets/models/angular_dense_2Apf_epoch_10_16451353003083222301") ==
            0
    @assert MLNanoShaperRunner.set_cutoff_radius(3.0f0) == 0

    file = getproperty.(
        read("$(homedir())/datasets/pqr/1/structure.pqr", PQR{Float32}), :pos)
    data = MLNanoShaperRunner.CSphere.(file)
    @assert MLNanoShaperRunner.load_atoms(pointer(data), length(data)) == 0
    (; x, y, z) = first(data)
    MLNanoShaperRunner.eval_model(x, y, z)
end
