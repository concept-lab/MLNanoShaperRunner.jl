using Test, ChainRulesTestUtils,Zygote
import FiniteDifferences
using MLNanoShaperRunner, CUDA

a = [1 2
     3 4]
@testset "batched_sum" begin
    @test begin
        batched_sum(a, [0, 1]) == [1; 3;;]
    end
    @test begin
        batched_sum(a, [1, 2]) == [2; 4;;]
    end
    @test begin
        batched_sum(a, [0, 2]) == [3; 7;;]
    end
    if CUDA.functional()
        @test begin
            batched_sum(cu(a), [0, 1]) == cu([1; 3;;])
        end
        @test begin
            batched_sum(cu(a), [1, 2]) == cu([2; 4;;])
        end
        @test begin
            batched_sum(cu(a), [0, 2]) == cu([3; 7;;])
        end
    end
end

@testset "derivations" begin
	@test begin
		jacobian(batched_sum,[1;;2],[0,2])[1] ≈ FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),a ->batched_sum(a,[0,2]),Float32.([1;;2]))[1]
	end
	@test begin
		jacobian(batched_sum,a,[0,2])[1] ≈ FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),a ->batched_sum(a,[0,2]),Float32.(a))[1]
	end
	@test begin
		jacobian(batched_sum,[1;;2],[0,1,2])[1] ≈ FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),a ->batched_sum(a,[0,1,2]),Float32.([1;;2]))[1]
	end
	@test begin
		jacobian(batched_sum,a,[0,2])[1] ≈ FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5, 1),a ->batched_sum(a,[0,2]),Float32.(a))[1]
	end
end
