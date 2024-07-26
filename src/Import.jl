module Import
using StructArrays
# using GLMakie
using GeometryBasics
using TOML
using BioStructures
using Folds
export extract_balls, PQR, Atom

struct XYZR{T} end
struct PQR{T} end
struct Atom{T}
    atom_number::Int64
    atom_name::Symbol
    residue_name::Symbol
    chain_id::Int64
    pos::Sphere{T}
    charge::T
end
base_type(::Type{XYZR{T}}) where {T} = Sphere{T}
base_type(::Type{PQR{T}}) where {T} = Atom{T}

function parse_line(line::String, ::Type{Atom{T}}) where {T}
    type, atom_number, atom_name, residue_name, chain_id, x, y, z, charge, r = split(line)
    atom_number, chain_id = parse.(Int64, (atom_number, chain_id))
    x, y, z, r, charge = parse.(T, (x, y, z, r, charge))
    if Symbol(type) == :ATOM
        Atom(atom_number,
            Symbol(atom_name),
            Symbol(residue_name),
            chain_id,
            Sphere(Point3(x, y, z), r),
            charge)
    else
        error("type not supported $type")
    end
end

function parse_line(line::String, ::Type{Sphere{T}}) where {T}
    x, y, z, r = parse.(T, split(line))
    Sphere(Point3(x, y, z), r)
end
function Base.read(io::IO, T::Type{<:Union{XYZR,PQR}})
    Folds.map(readlines(io)) do line
        parse_line(line, base_type(T))
    end
end

# function viz(x::AbstractArray{Sphere{T}}) where {T}
#     fig = Figure()
#     ax = Axis3(fig[1, 1])
#     mesh!.(Ref(ax), x)
#     fig
# end

reduce(fun, arg) = mapreduce(fun, vcat, arg)
reduce(fun) = arg -> reduce(fun, arg)
function reduce(fun, arg, n::Integer)
    if n <= 1
        reduce(fun, arg)
    else
        reduce(reduce(fun), arg, n - 1)
    end
end

function export_file(io::IO, prot::AbstractArray{Sphere{T}}) where {T}
    for sph in prot
        println(io, sph.center[1], " ", sph.center[2], " ", sph.center[3], " ", sph.r)
    end
end

params = "$( dirname(dirname(@__FILE__)))/param/param.toml"

function extract_balls(T::Type{<:Number}, prot::MolecularStructure)
    radii = TOML.parsefile(params)["atoms"]["radius"] |> Dict{String,T}
    reduce(prot, 4) do atom
        if typeof(atom) == Atom
            Sphere{T}[Sphere(Point3(atom.coords) .|> T,
                if atom.element in keys(radii)
                    radii[atom.element]
                else
                    1.0
                end)]
        else
            Sphere{T}[]
        end
    end |> StructVector
end
end
