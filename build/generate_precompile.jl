using MLNanoShaperRunner


function basic_inference()
    MLNanoShaperRunner.load_weights("$(homedir())/datadir/models")
    MLNanoShaperRunner.set_cutoff_radius(3.0)

    file = load_pqr("$(homedir())/datadir/pqr/1")
    MLNanoShaperRunner.load_atoms(MLNanoShaperRunner.CSphere.(file))
    MLNanoShaperRunner.eval_model(1.0f0, 0.0f0, 3.0f0)
end
