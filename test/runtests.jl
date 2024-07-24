using Test
using MLNanoShaperRunner, CUDA

@testset "batchsum" begin
    a = [1 2
         3 4]
    @test begin
		batchsum(a, [0, 1]) == [1; 3]
    end
    @test begin
        batchsum(a, [1, 2]) == [2; 4]
    end
    @test begin
        batchsum(a, [0, 2]) == [3; 7]
    end
    @test begin
		batchsum(cu(a), [0, 1]) == cu([1; 3])
    end
    @test begin
		batchsum(cu(a), [1, 2]) == cu([2; 4])
    end
    @test begin
		batchsum(cu(a), [0, 2]) == cu([3; 7])
    end
end
