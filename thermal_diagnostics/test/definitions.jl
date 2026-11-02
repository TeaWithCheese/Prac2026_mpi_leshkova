using Test
using thermal_diagnostics.Definitions

@testset "Definitions" begin

    @testset "PhysicsParam" begin
        p = PhysicsParam()
        @test p.L == 1.0
        @test p.H == 0.3
        @test p.b == 1.0
        @test p.c_s == 1.0
        @test p.k_s == 0.1
        @test p.h_a == 0.03
        @test p.A_c == 0.05
        @test p.c_c == 1.0
        @test p.β == 0.1

        p2 = PhysicsParam(k_s = 0.2, β = 0.15)
        @test p2.k_s == 0.2
        @test p2.β == 0.15
        @test p2.L == 1.0
    end

    @testset "Counts" begin
        c = Counts(20, 6, 17)
        @test c.Nx == 20
        @test c.Ny == 6
        @test c.Ns == 120
        @test c.Nc == 17
        @test c.Nz == 137
    end

    @testset "GridBody" begin
        p = PhysicsParam()
        c = Counts(20, 6, 17)
        gb = GridBody(p, c)

        @test gb.Nx == 20
        @test gb.Ny == 6
        @test gb.Ns == 120
        @test gb.dx == p.L / c.Nx
        @test gb.dy == p.H / c.Ny
        @test length(gb.xc) == c.Nx
        @test length(gb.yc) == c.Ny
        @test gb.Vs ≈ p.b * gb.dx * gb.dy

        # центры ячеек
        @test gb.xc[1] ≈ gb.dx / 2
        @test gb.xc[end] ≈ p.L - gb.dx / 2
        @test gb.yc[1] ≈ gb.dy / 2
        @test gb.yc[end] ≈ p.H - gb.dy / 2
    end

    @testset "GridChannel" begin
        p = PhysicsParam()
        c = Counts(20, 6, 17)
        gc = GridChannel(p, c)

        @test gc.Nc == 17
        @test length(gc.sb) == c.Nc + 1
        @test length(gc.sc) == c.Nc
        @test length(gc.ds) == c.Nc
        @test gc.sb[1] == 0.0
        @test gc.sb[end] == p.L
        @test sum(gc.ds) ≈ p.L

        # центры — середины отрезков, длины — разности границ
        for j in 1:c.Nc
            @test gc.sc[j] == (gc.sb[j] + gc.sb[j + 1]) / 2
            @test gc.ds[j] == gc.sb[j + 1] - gc.sb[j]
            @test gc.ds[j] > 0
        end
    end

    @testset "Source" begin
        src = Source((x, y, t) -> x + 10y + 100t, t -> 0.0)
        @test src.q(0.5, 0.1, 0.01) == 2.5
        @test src.q(0.0, 0.0, 0.0) == 0.0
        @test src.Ta(1.0) == 0.0
    end

    @testset "Body" begin
        bd = Body((x, y) -> x + 10y)
        @test bd.Ts0(0.5, 0.1) == 1.5
        @test bd.Ts0(0.1, 0.5) == 5.1
    end

    @testset "FlowChannel" begin
        ch = FlowChannel(s -> 2s, t -> 3t)
        @test ch.Tc0(0.5) == 1.0
        @test ch.Tc0(1.0) == 2.0
        @test ch.Tin(2.0) == 6.0
    end

    @testset "Boundary" begin
        p = PhysicsParam()
        c = Counts(20, 6, 17)
        gb = GridBody(p, c)
        gc = GridChannel(p, c)

        segments = [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)]
        h = [2.0, 2.0, 2.0]
        bnd = Boundary(gb, gc, segments, h)

        @test bnd.M == 3
        @test bnd.segments == segments
        @test bnd.h == h
        @test !isempty(bnd.ell)

        # каждая тройка имеет положительную длину
        for ((i, j, m), ℓ) in bnd.ell
            @test ℓ > 0
            @test 1 <= i <= gb.Nx
            @test 1 <= j <= gc.Nc
            @test 1 <= m <= bnd.M
        end

        # суммы пересечений по каждому отрезку равны его длине
        for m in 1:bnd.M
            xL, xR = segments[m]
            total = sum(ℓ for ((_, _, mm), ℓ) in bnd.ell if mm == m)
            @test total ≈ (xR - xL)
        end
    end

    @testset "Boundary — ошибки" begin
        p = PhysicsParam()
        c = Counts(20, 6, 17)
        gb = GridBody(p, c)
        gc = GridChannel(p, c)

        # пустой список отрезков
        @test_throws ArgumentError Boundary(gb, gc,
            Tuple{Float64,Float64}[], Float64[])

        # длины segments и h не совпадают
        @test_throws ArgumentError Boundary(gb, gc,
            [(0.0, 0.5), (0.5, 1.0)], [2.0])

        # неположительный h
        @test_throws ArgumentError Boundary(gb, gc,
            [(0.0, 0.5), (0.5, 1.0)], [2.0, 0.0])
        @test_throws ArgumentError Boundary(gb, gc,
            [(0.0, 0.5), (0.5, 1.0)], [-1.0, 2.0])

        # отрезок выходит за [0, L]: сумма пересечений не совпадёт с длиной
        @test_throws ArgumentError Boundary(gb, gc,
            [(0.0, 0.5), (0.5, 1.5)], [2.0, 2.0])
    end
end
