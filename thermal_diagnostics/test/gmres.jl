using Test, thermal_diagnostics
import LinearAlgebra, Random

@testset "Parallel vector operations for GMRES" begin

test_vectors = [
    (Vector{Float64}(), Vector{Float64}()),
    ([1e11, 1e11], [1e11, -1e11]),
    (rand(Random.Xoshiro("x16"), 16), rand(Random.Xoshiro("y16"), 16)),
    (rand(Random.Xoshiro("x"), 10000), rand(Random.Xoshiro("y"), 10000)),
]
@testset "Parallel vector operations n=$(length(x))" for (x, y) in test_vectors
    @test thermal_diagnostics.dot(x, y) ≈ LinearAlgebra.dot(x, y) rtol=1e-10
    @test thermal_diagnostics.norm(x) ≈ LinearAlgebra.norm(x) rtol=1e-10
    @test thermal_diagnostics.mul!(y, x, 0.5) == x .* 0.5
    @test thermal_diagnostics.muladd!(copy(y), x, 2.) == y .+ x .* 2
end

end # @testset "Parallel vector operations for GMRES"

@testset "GMRES" begin

@testset "GMRES edge cases" begin
    @test thermal_diagnostics.givens_elimination(0, 0) ==
        (thermal_diagnostics.GivensRotation(1, 0), 0)
    matvec = x -> [1. 0.; 0. 0.] * x
    z = zeros(2)
    @test gmres(matvec, [1., 0.], z) == [1., 0.]
    @test_throws LinearAlgebra.SingularException gmres(matvec, [1., 1.], z)

    matvec = x -> [0. 1.; 1. 0.] * x
    b = [1., 0.]
    @test gmres(matvec, b, z; maxiter=2) == [0., 1.]
    try
        gmres(matvec, b, z; maxiter=1)
    catch e
        @test e isa thermal_diagnostics.ConvergenceException
        @test iszero(e.lastapprox)
    end
end

function diagonally_dominant(n::Int, rng=Random.default_rng(),)
    A = rand(rng, n, n)
    for j = 1:n
        A[j, j] += sum(@view A[:, j])
    end
    A
end
@testset "GMRES for diagonally dominant n=$n" for n in 4 .^ (1:6)
    rng = Random.Xoshiro("gmres$n")
    A = diagonally_dominant(n, rng)
    b = rand(rng, n)
    x = gmres(x -> A * x, b, zeros(n))
    @test LinearAlgebra.norm(A * x .- b) <= 1e-10 * LinearAlgebra.norm(b)
end

end # @testset "GMRES"
