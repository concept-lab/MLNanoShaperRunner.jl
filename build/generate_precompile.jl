using MLNanoShaperRunner
using StructArrays


function basic_inference()
    MLNanoShaperRunner.load_weights("$(homedir())/datasets/models/angular_dense_2An_epoch_50_6194726998944467677")
    MLNanoShaperRunner.set_cutoff_radius(3.0f0)

	file = getproperty.(read("$(homedir())/datasets/pqr/1/structure.pqr",PQR{Float32}),:pos) 
	data = MLNanoShaperRunner.CSphere.(file)
	@info "data" data
	MLNanoShaperRunner.load_atoms(pointer_from_objref(data) |> Ptr{MLNanoShaperRunner.CSphere},size(data) )
    MLNanoShaperRunner.eval_model(1.0f0, 0.0f0, 3.0f0)
end
