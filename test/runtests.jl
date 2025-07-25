using Test, ChainRulesTestUtils, Zygote
import FiniteDifferences
using MLNanoShaperRunner, CUDA
using MLNanoShaperRunner.Import: PQR
using MLNanoShaperRunner: batched_sum 
using StructArrays,Serialization,GeometryBasics
using Lux

a = [1 2
     3 4]
@testset "batched_sum" begin
    @test begin
        batched_sum(a, 1,[0, 1]) == [1; 3;;]
    end
    @test begin
        batched_sum(a,1, [1, 2]) == [2; 4;;]
    end
    @test begin
        batched_sum(a,2, [0, 2]) == [3; 7;;]
    end
    if CUDA.functional()
        @test begin
            batched_sum(cu(a),1, [0, 1]) == cu([1; 3;;])
        end
        @test begin
            batched_sum(cu(a),1, [1, 2]) == cu([2; 4;;])
        end
        @test begin
            batched_sum(cu(a),2, [0, 2]) == cu([3; 7;;])
        end
    end
end

@testset "derivations" begin
    @test begin
        jacobian(batched_sum, [1;; 2],2, [0, 2])[1] ≈
        FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),
            a -> batched_sum(a,2, [0, 2]), Float32.([1;; 2]))[1]
    end
    @test begin
        jacobian(batched_sum, a,2, [0, 2])[1] ≈ FiniteDifferences.jacobian(
            FiniteDifferences.central_fdm(5, 1), a -> batched_sum(a,2, [0, 2]), Float32.(a))[1]
    end
    @test begin
        jacobian(batched_sum, [1;; 2],1, [0, 1, 2])[1] ≈
        FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),
            a -> batched_sum(a,1, [0, 1, 2]), Float32.([1;; 2]))[1]
    end
    @test begin
        jacobian(batched_sum, a,2, [0, 2])[1] ≈ FiniteDifferences.jacobian(
            FiniteDifferences.central_fdm(5, 1), a -> batched_sum(a,2, [0, 2]), Float32.(a))[1]
    end
    @test begin
        jacobian(batched_sum, cu([1;; 2]), 2,[0, 2])[1] |> Array ≈
        FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),
            a -> batched_sum(a,2, [0, 2]), Float32.([1;; 2]))[1]
    end
    @test begin
        jacobian(batched_sum, cu(a),2, [0, 2])[1] |> Array ≈ FiniteDifferences.jacobian(
            FiniteDifferences.central_fdm(5, 1), a -> batched_sum(a,2, [0, 2]), Float32.(a))[1]
    end
    @test begin
        jacobian(batched_sum, cu([1;; 2]),1, [0, 1, 2])[1] |> Array ≈
        FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),
            a -> batched_sum(a,1, [0, 1, 2]), Float32.([1;; 2]))[1]
    end
    @test begin
        jacobian(batched_sum, cu(a),2, [0, 2])[1] |> Array ≈ FiniteDifferences.jacobian(
            FiniteDifferences.central_fdm(5, 1), a -> batched_sum(a,2, [0, 2]), Float32.(a))[1]
    end
end
function provide_start_model(model::Lux.StatefulLuxLayer,i::Integer)::StatefulLuxLayer
    Lux.StatefulLuxLayer{true}(provide_start_model(model.model,i),model.ps,model.st)
end
function provide_start_model(model::Lux.Chain,i::Integer)::Chain
    Lux.Chain([model[k] for k in 1:i]...,[NoOpLayer() for _ in (i+1):length(model)]...)
end
function provide_start_secondary_chain(model::Lux.StatefulLuxLayer,i::Integer)::StatefulLuxLayer
    n = length(model.model)
    Lux.StatefulLuxLayer{true}(Lux.Chain([model.model[k] for k in 1:(n-1)]...,provide_start_model(model.model[n],i)),model.ps,model.st)
end
@testset "preprocessing" begin
    atoms1 = StructVector([Sphere(Point3f(0,0,0),1f0),Sphere(Point3f(0,-1,0),2f0)])
    atoms2 = StructVector([Sphere(Point3f(0,0,0),.5f0),Sphere(Point3f(.3,0,0),1f0)])
    points = [Point3f(0,1,0),Point3f(0,0,1)]
    cutoff_radius=3f0
    @test MLNanoShaperRunner.preprocessing(Batch(points),Batch([atoms1,atoms1]);cutoff_radius).field  ≈ MLNanoShaperRunner.preprocessing(Batch(cu(points)),Batch([atoms1,atoms1]);cutoff_radius).field |> Array
    @test MLNanoShaperRunner.preprocessing(Batch(points[1:1]),Batch([atoms2]);cutoff_radius).field  ≈ MLNanoShaperRunner.preprocessing(Batch(cu(points[1:1])),Batch([atoms2]);cutoff_radius).field |> Array
end
@testset "evaluation" begin
    model_file = "$(@__DIR__)/../examples/tiny_angular_dense_s_final_training_10_3.0_categorical_6000_6331735514142882335"
    protein_file = "$(@__DIR__)/../examples/example_1.pqr"
    protein = RegularGrid(getfield.(read(protein_file, PQR{Float32}), :pos) |> StructVector,3f0)
    model = production_instantiate(deserialize(model_file))
    preprocessing_layer = Lux.StatefulLuxLayer{true}(MLNanoShaperRunner.get_preprocessing(model.model),model.ps,model.st)

    @test begin
         a1::Float32 = model((Point3f(0,22,10),protein)) |> only
         a2::Float32,_  = model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein)) |> vec
         a1 ≈ a2
    end
    @test begin
         b1::Float32 = model((Point3f(0,22,11),protein)) |> only
         _,b2::Float32 = model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein)) |> vec
         b1 ≈ b2
    end

    @test begin
         a1 = MLNanoShaperRunner.get_element(preprocessing_layer((Point3f(0,22,10),protein)),1)
         a2 = MLNanoShaperRunner.get_element(preprocessing_layer((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein)),1) 
         a1 ≈ a2
    end
    @test begin
         b1= MLNanoShaperRunner.get_element(preprocessing_layer((Point3f(0,22,11),protein)),1)
         b2=MLNanoShaperRunner.get_element(preprocessing_layer((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein)),2) 
         b1 ≈ b2
    end
    @test begin
         a1 = preprocessing_layer((Point3f(0,22,10),protein)).lengths
         a2 = preprocessing_layer((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein)).lengths 
         a1[2] == a2[2]
    end
    @test begin
         b1= preprocessing_layer((Point3f(0,22,11),protein)).lengths
         b2=preprocessing_layer((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein)).lengths 
         b1[2] == b2[3] - b2[2]
    end

    @test let start_model = provide_start_model(model,2)
         a1= start_model((Point3f(0,22,10),protein))[:,1]
         a2  = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,1] 
         a1 ≈ a2
    end
    @test let start_model = provide_start_model(model,2)
         b1 = start_model((Point3f(0,22,11),protein))[:,1]
         b2 = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,2] 
         b1 ≈ b2
    end
    @test let start_model = provide_start_model(model,3)
         a1= start_model((Point3f(0,22,10),protein))[:,1]
         a2  = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,1] 
         a1 ≈ a2
    end
    @test let start_model = provide_start_model(model,3)
         b1 = start_model((Point3f(0,22,11),protein))[:,1]
         b2 = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,2] 
         b1 ≈ b2
    end

    @test let start_model = provide_start_secondary_chain(model,1)
         a1= start_model((Point3f(0,22,10),protein))[:,1]
         a2  = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,1] 
         a1 ≈ a2
    end
    @test let start_model = provide_start_secondary_chain(model,1)
         b1 = start_model((Point3f(0,22,11),protein))[:,1]
         b2 = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,2] 
         b1 ≈ b2
    end

    @test let start_model = provide_start_secondary_chain(model,2)
         a1= start_model((Point3f(0,22,10),protein))[:,1]
         a2  = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,1] 
         a1 ≈ a2
    end
    @test let start_model = provide_start_secondary_chain(model,2)
         b1 = start_model((Point3f(0,22,11),protein))[:,1]
         b2 = start_model((MLNanoShaperRunner.Batch([Point3f(0,22,10),Point3f(0,22,11)]),protein))[:,2] 
         b1 ≈ b2
    end

end
