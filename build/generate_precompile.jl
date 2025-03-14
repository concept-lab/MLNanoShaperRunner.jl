using MLNanoShaperRunner
using StructArrays

function basic_inference()
    @assert MLNanoShaperRunner.load_model(Base.unsafe_convert(Cstring,"$(@__DIR__)/../examples/tiny_angular_dense_s_jobs_11_6_3_c_2025-03-10_epoch_800_10631177997949843226")) == 0
    file = getproperty.(
        read("$(homedir())/datasets/pqr/1/structure.pqr", MLNanoShaperRunner.PQR{Float32}), :pos)
    data = MLNanoShaperRunner.CSphere.(file)
    @assert MLNanoShaperRunner.load_atoms(pointer(data), length(data) |> Cint) == 0
    (; x, y, z) = first(data)
    array = [MLNanoShaperRunner.CPoint(x,y,z)]
    MLNanoShaperRunner.eval_model(pointer(array), 1 |> Cint)
end
basic_inference()
