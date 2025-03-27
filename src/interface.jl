function evaluate_model(model::StatefulLuxLayer{true},atoms::RegularGrid;step::Number=1)::Array{Float32,3}
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
	volume = Folds.map(grid) do x
        model((Batch([x]), atoms)) |> only
    end
	volume
end
