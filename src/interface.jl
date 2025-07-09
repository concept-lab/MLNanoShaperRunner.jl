function count_point_types(coordinates::AbstractArray{Point3f,3},atoms::RegularGrid)
	cutoff_radius = atoms.radius
	cutoff_radius² = cutoff_radius^2
	local_unknown_counts=zeros(Int,Threads.nthreads())
	local_known_counts=zeros(Int,Threads.nthreads())
	local_outside_counts=zeros(Int,Threads.nthreads())
	@info length(coordinates)
    @threads for I in eachindex(IndexCartesian(), coordinates)
    	thread_id = Threads.threadid()
		i = Tuple(I)
		pos = coordinates[I]
		has_atoms_nearby = Ref(false)
		is_inside_atom = Ref(false)
		_iter_grid(atoms,pos,Δ3) do s::Sphere{Float32}
			d² = (s.center.- pos ) .^2 |> sum
			if d² < s.r^2
				is_inside_atom[] = true
				return true
			elseif d² < cutoff_radius²
				has_atoms_nearby[] = true
			end
			return false
		end
		if is_inside_atom[]
			local_known_counts[thread_id] +=1
		elseif has_atoms_nearby[]
			# @info "unknown indices" I
			local_unknown_counts[thread_id]+=1
		else
			local_outside_counts[thread_id]+=1
		end
	end
	(;known_counts = sum(local_known_counts),unknown_counts = sum(local_unknown_counts),outside_counts = sum(local_outside_counts))
end
function evaluate_trivial!(volume::AbstractArray{Float32,3},coordinates::AbstractArray{Point3f,3},atoms::RegularGrid)::Tuple{AbstractVector{CartesianIndex{3}},AbstractVector{Point3f}}
	cutoff_radius = atoms.radius
	cutoff_radius² = cutoff_radius^2
	n = length(volume)
	k = Threads.nthreads()
	vec_unknow_indices = Matrix{CartesianIndex{3}}(undef,k,n ÷ k)
	vec_unknow_pos = Matrix{Point3f}(undef,k,n ÷ k) 
	id_last = zeros(Int,k)
	has_atoms_nearby = falses(k)
    for I in eachindex(IndexCartesian(), coordinates)
		pos = coordinates[I]
    	k = Threads.threadid()
		has_atoms_nearby[k] = false
		volume[I] = 0f0
		_iter_grid(atoms,pos,Δ3) do s::Sphere{Float32}
			d² = (s.center.- pos ) .^2 |> sum
			if d² < s.r^2
				volume[I] = 1f0
				has_atoms_nearby[k] = false
				return true
			elseif d² < cutoff_radius² && volume[I] == 0f0
				# @assert volume[I] == 0f0 "got $(volume[I])"
				has_atoms_nearby[k] = true
			end
			return false
		end
		if has_atoms_nearby[k]
			# @assert volume[I] == 0f0 "got $(volume[I])"
			id_last[k] += 1
			vec_unknow_indices[k,id_last[k]]= I
			vec_unknow_pos[k,id_last[k]]  = pos
		end
	end
	unknown_indices = CartesianIndex{3}[]
	unknown_pos = Point3f[]
	# for k in 1:Threads.nthreads()
		# append!(unknown_indices,view(vec_unknow_indices[k],1:id_last[k]))
		# append!(unknown_pos,view(vec_unknow_pos[k],1:id_last[k]))
	# end
	unknown_indices,unknown_pos
end
function coord_to_pos(mins,step,coord)
	mins + step * (coord - 1)
end
function pos_to_coord(mins,step,pos)
	(pos - mins) / step + 1
end
function iter_coordinates(mins::Point3f,sizes::NTuple{3,Int},step::Float32,pos::Point3f,radius::Float32)
	search_min = max.(floor.(Int,pos_to_coord.(mins,step,pos .- radius)), 1) |> Tuple
	search_max = min.(ceil.( Int,pos_to_coord.(mins,step,pos.+ radius)), sizes) |> Tuple
	CartesianIndices{3,Tuple{UnitRange{Int64}, UnitRange{Int64}, UnitRange{Int64}}}(range.(search_min, search_max))
end
function evaluate_trivial_fast!(volume::AbstractArray{Float32,3}, mins, step, atoms::AbstractVector{Sphere{Float32}})::AbstractVector{CartesianIndex{3}}
   for (;center,r) in atoms
   		r² = r^2
   		R² = (r + 1.4f0)^2
   		# cutoff_radius² = atoms.radius^2
    	for I in iter_coordinates(mins,size(volume),step,center,r + 1.4f0) 
			pos = mins .+ step .* (Tuple(I) .- 1)
			d² = (center .- pos) .^ 2 |> sum
			if d² < r²
    			volume[I] = 1f0
    		elseif d² < R² && volume[I] < 1f0
    			volume[I] = .5f0
    		end

		end
	end
	unknown_indices = CartesianIndex{3}[]
	for (i,v) in pairs(IndexCartesian(),volume)
		if v == .5f0
			push!(unknown_indices,i)
		end
	end
	unknown_indices
end
@inbounds function evaluate_field_fast(model::StatefulLuxLayer, atoms::StructVector{Sphere{Float32}}; step::Number=1.0f0, batch_size=100000)#::Array{Float32,3}
	_atoms = RegularGrid(atoms,get_cutoff_radius(model.model))
	mins = _atoms.start .- 2
	maxes = mins .+ size(_atoms.grid) .* _atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    volume = similar(grid,Float32)
    unknown_indices= evaluate_trivial_fast!(volume,mins,step,atoms)
    unknown_pos::Vector{Point3f} = map(unknown_indices) do i coord_to_pos.(mins,step,Point3f(i[1],i[2],i[3])) end
    # @info "comparing lengths" length(unknown_indices)/batch_size 
    for i in 1:batch_size:length(unknown_indices)
    	k = min(i+ batch_size-1,length(unknown_indices))
    	p=  view(unknown_pos,i:k) |> Batch
    	r = model((p, _atoms))
    	res::Vector{Float32} = r |> cpu_device() |> vec
    	# @assert length(res) == length(v)
    	for (l,r) in zip(i:k,res)
	    	volume[unknown_indices[l]] = r 
    	end
    end
	volume
end
function evaluate_field(model::StatefulLuxLayer,atoms::RegularGrid;step::Number=1,batch_size = 100000)::Array{Float32,3}
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    g = vec(grid)
    volume = similar(grid,Float32)
    v = reshape(volume,:)
    @info "comparing lengths" length(volume)/batch_size 
    for i in 1:batch_size:length(volume)
    	k = min(i+ batch_size-1,length(v))
    	res =  model((Batch(view(g,i:k)), atoms)) |> cpu_device() |> vec
    	v[i:k] .= res
        end
	volume
end
