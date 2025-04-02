function evaluate_field(model::StatefulLuxLayer,atoms::RegularGrid;step::Number=1)::Array{Float32,3}
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    g = reshape(grid,:)
    batch_size = 2
    volume = similar(grid,Float32)
    v = reshape(volume,:)
    for i in 1:batch_size:length(volume)
    	k = min(i+ batch_size-1,length(v))
    	res =  reshape(model((Batch(view(g,i:k)), atoms)),:)
    	v[i:k] .= res
        end
	volume
end
