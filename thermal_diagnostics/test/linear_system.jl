using Test
using thermal_diagnostics.LinearSystem

@testset "LinearSystem_matvecs" begin

    @testset "M_matvec" begin
        p  = PhysicsParam(H = 0.5)
        c  = Counts(2, 2, 2)
        gb = GridBody(p, c)
        gc = GridChannel(p, c, 1.0)

        @test gb.dx == 0.5
        @test gb.dy == 0.25
        @test gb.Vs == 0.125
        @test gc.ds == [0.5, 0.5]

        @testset "M * ones" begin
            res = M_matvec(ones(c.Nz), gb.Vs, p, gc)
            @test res ≈ [0.125, 0.125, 0.125, 0.125, 0.025, 0.025]
        end

        @testset "M * (1, 2, 3, 4, 5, 6)" begin
            v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            res = M_matvec(v, gb.Vs, p, gc)
            @test res ≈ [0.125, 0.25, 0.375, 0.5, 0.125, 0.15]
        end

        @testset "linearity" begin
            v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            w = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
            @test M_matvec(v .+ w, gb.Vs, p, gc) ≈
                  M_matvec(v, gb.Vs, p, gc) .+ M_matvec(w, gb.Vs, p, gc)
        end
    end

    @testset "K_matvec" begin
        p  = PhysicsParam(H = 0.5)
        c  = Counts(2, 2, 2)
        gb = GridBody(p, c)
        gc = GridChannel(p, c, 1.0)
        bnd = Boundary(gb, gc, [(0.0, 0.5), (0.5, 1.0)], [1.0, 3.0])

        cond = Conductivity(p, gb)

        @testset "conductances" begin
            @test cond.gx  ≈ 0.05
            @test cond.gy  ≈ 0.2
            @test cond.gxa ≈ 0.0075
            @test cond.gya ≈ 0.015
        end

        @testset "g_ijm" begin
            @test g_ijm(1, 1, 1, p.b, bnd) ≈ 0.5
            @test g_ijm(2, 2, 2, p.b, bnd) ≈ 1.5
            @test g_ijm(1, 2, 1, p.b, bnd) ≈ 0.0
            @test g_ijm(2, 1, 2, p.b, bnd) ≈ 0.0
            @test g_ijm(1, 1, 2, p.b, bnd) ≈ 0.0
            @test g_ijm(2, 2, 1, p.b, bnd) ≈ 0.0
        end

        @testset "K * ones" begin
            res = K_matvec(ones(c.Nz), c, bnd, p, gb)
            @test res ≈ [0.0075, 0.0075, 0.0225, 0.0225, 0.1, 0.0] atol=1e-14
        end

        @testset "K * (1, 2, 3, 4, 5, 6)" begin
            v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            res = K_matvec(v, c, bnd, p, gb)
            expected = [-2.4425, -6.335, 0.4175, 0.54, 2.5, 6.1]
            @test res ≈ expected atol=1e-14
        end

        @testset "K * (-1, 1, -1, 1, -1, 1)" begin
            v = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
            res = K_matvec(v, c, bnd, p, gb)
            expected = [-0.1075, 0.1075, -0.1225, 0.1225, -0.1, 0.2]
            @test res ≈ expected atol=1e-14
        end

        @testset "linearity" begin
            v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            w = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
            @test K_matvec(v .+ w, c, bnd, p, gb) ≈
                  K_matvec(v, c, bnd, p, gb) .+ K_matvec(w, c, bnd, p, gb)
        end

        @testset "K_s row sums are zero" begin
            for i in 1:c.Nx, k in 1:c.Ny
                a = body_index(i, k, c)
                s = 0.0
                for b in 1:c.Ns
                    s += K_s_ab(a, b, cond.gx, cond.gy, c)
                end
                @test s ≈ 0.0 atol=1e-14
            end
        end

        @testset "K_Г row sums are zero" begin
            for a in 1:c.Nz
                s = 0.0
                for b in 1:c.Nz
                    s += K_Г_ab(a, b, c, p.b, bnd)
                end
                @test s ≈ 0.0 atol=1e-14
            end
        end

        @testset "K_s + K_a + K_Г is symmetric" begin
            M_sym = zeros(c.Nz, c.Nz)
            for a in 1:c.Nz, b in 1:c.Nz
                M_sym[a, b] = K_s_ab(a, b, cond.gx, cond.gy, c) +
                              K_a_ab(a, b, cond.gxa, cond.gya, c) +
                              K_Г_ab(a, b, c, p.b, bnd)
            end
            @test M_sym ≈ transpose(M_sym) atol=1e-14
        end
    end

    @testset "block zeroing" begin
        c  = Counts(2, 2, 2)
        p  = PhysicsParam(H = 0.5)
        gb = GridBody(p, c)
        gc = GridChannel(p, c, 1.0)
        bnd = Boundary(gb, gc, [(0.0, 0.5), (0.5, 1.0)], [1.0, 3.0])

        @testset "h_a = 0 → K_a = 0" begin
            p0 = PhysicsParam(H = 0.5, h_a = 0.0)
            cond0 = Conductivity(p0, gb)
            for a in 1:c.Nz, b in 1:c.Nz
                @test K_a_ab(a, b, cond0.gxa, cond0.gya, c) == 0.0
            end
        end

        @testset "b_body = 0 → K_Г = 0" begin
            for a in 1:c.Nz, b in 1:c.Nz
                @test K_Г_ab(a, b, c, 0.0, bnd) == 0.0
            end
        end

        @testset "β = 0 → K_c = 0" begin
            for a in 1:c.Nz, b in 1:c.Nz
                @test K_c_ab(a, b, c, 0.0) == 0.0
            end
        end
    end

        @testset "K_Г_ab with multiple m per (i,j)" begin
        p = PhysicsParam(H = 0.5)
        c = Counts(2, 1, 2)
        gb = GridBody(p, c)
        gc = GridChannel(p, c, 1.0)
        # сегменты рассекают ячейку i=2 канала j=2 на две части
        bnd = Boundary(gb, gc, [(0.0, 0.6), (0.6, 1.0)], [1.0, 2.0])

        # (i=2, j=2) пересекается и с m=1 (0.1), и с m=2 (0.4)
        @test g_ijm(2, 2, 1, p.b, bnd) ≈ 0.1
        @test g_ijm(2, 2, 2, p.b, bnd) ≈ 0.8
        @test g_ijm(1, 1, 1, p.b, bnd) ≈ 0.5
        @test g_ijm(1, 1, 2, p.b, bnd) ≈ 0.0
        @test g_ijm(2, 1, 1, p.b, bnd) ≈ 0.0
        @test g_ijm(2, 1, 2, p.b, bnd) ≈ 0.0
        @test g_ijm(1, 2, 1, p.b, bnd) ≈ 0.0
        @test g_ijm(1, 2, 2, p.b, bnd) ≈ 0.0

        # диагональные элементы: сумма g_ijm по всем m для каждой (i,j)
        @test K_Г_ab(1, 1, c, p.b, bnd) ≈ 0.5
        @test K_Г_ab(2, 2, c, p.b, bnd) ≈ 0.9    # 0.1 + 0.8
        @test K_Г_ab(3, 3, c, p.b, bnd) ≈ 0.5    # канал j=1
        @test K_Г_ab(4, 4, c, p.b, bnd) ≈ 0.9    # канал j=2

        # недиагональные: деталь a=2 (i=2) ↔ канал a=4 (j=2)
        @test K_Г_ab(2, 4, c, p.b, bnd) ≈ -0.9
        @test K_Г_ab(4, 2, c, p.b, bnd) ≈ -0.9

        # деталь a=1 (i=1) не пересекается с j=2
        @test K_Г_ab(1, 4, c, p.b, bnd) ≈ 0.0
        @test K_Г_ab(4, 1, c, p.b, bnd) ≈ 0.0
    end

    @testset "single-cell body and channel (Nx=Ny=Nc=1)" begin
        p = PhysicsParam(H = 0.5)
        c = Counts(1, 1, 1)
        gb = GridBody(p, c)
        gc = GridChannel(p, c, 1.0)
        bnd = Boundary(gb, gc, [(0.0, 1.0)], [1.0])
        cond = Conductivity(p, gb)

        @test gb.dx == 1.0
        @test gb.dy == 0.5
        @test gc.ds == [1.0]

        # одна ячейка детали — соседей нет
        @test K_s_ab(1, 1, cond.gx, cond.gy, c) == 0.0

        # внешняя теплоотдача: одна ячейка на всех 3 сторонах
        @test K_a_ab(1, 1, cond.gxa, cond.gya, c) ≈ cond.gxa + cond.gya

        # контакт: одна ячейка детали — одна ячейка канала, весь контакт
        @test g_ijm(1, 1, 1, p.b, bnd) ≈ 1.0
        @test K_Г_ab(1, 1, c, p.b, bnd) ≈ 1.0
        @test K_Г_ab(2, 2, c, p.b, bnd) ≈ 1.0
        @test K_Г_ab(1, 2, c, p.b, bnd) ≈ -1.0
        @test K_Г_ab(2, 1, c, p.b, bnd) ≈ -1.0

        # K_c: одна ячейка канала, диагональ = β, вход уходит в f
        @test K_c_ab(2, 2, c, p.β) ≈ p.β
        @test K_c_ab(1, 1, c, p.β) == 0.0
    end

    @testset "single row of body cells (Ny=1)" begin
        p = PhysicsParam(H = 0.5)
        c = Counts(3, 1, 2)
        gb = GridBody(p, c)
        cond = Conductivity(p, gb)

        # диагональ K_s: на краях по одному соседу, в середине — два
        @test K_s_ab(1, 1, cond.gx, cond.gy, c) ≈ cond.gx
        @test K_s_ab(2, 2, cond.gx, cond.gy, c) ≈ 2 * cond.gx
        @test K_s_ab(3, 3, cond.gx, cond.gy, c) ≈ cond.gx

        # недиагональ: только между реальными соседями по x
        @test K_s_ab(1, 2, cond.gx, cond.gy, c) ≈ -cond.gx
        @test K_s_ab(2, 1, cond.gx, cond.gy, c) ≈ -cond.gx
        @test K_s_ab(2, 3, cond.gx, cond.gy, c) ≈ -cond.gx
        @test K_s_ab(3, 2, cond.gx, cond.gy, c) ≈ -cond.gx
        # 1 и 3 не соседи (перепрыгивают через 2 и Nx=3)
        @test K_s_ab(1, 3, cond.gx, cond.gy, c) == 0.0
        @test K_s_ab(3, 1, cond.gx, cond.gy, c) == 0.0

        # K_a: k=Ny=1 у всех, но i=1 и i=3 ещё и на боках
        @test K_a_ab(1, 1, cond.gxa, cond.gya, c) ≈ cond.gxa + cond.gya
        @test K_a_ab(2, 2, cond.gxa, cond.gya, c) ≈ cond.gya
        @test K_a_ab(3, 3, cond.gxa, cond.gya, c) ≈ cond.gxa + cond.gya
    end

    @testset "single column of body cells (Nx=1)" begin
        p = PhysicsParam(H = 0.5)
        c = Counts(1, 3, 2)
        gb = GridBody(p, c)
        cond = Conductivity(p, gb)

        # диагональ K_s: крайние по одному, средний два
        @test K_s_ab(1, 1, cond.gx, cond.gy, c) ≈ cond.gy
        @test K_s_ab(2, 2, cond.gx, cond.gy, c) ≈ 2 * cond.gy
        @test K_s_ab(3, 3, cond.gx, cond.gy, c) ≈ cond.gy

        # в одном столбце горизонтального соседа нет: |a-b|=1 не даёт -gx
        @test K_s_ab(1, 2, cond.gx, cond.gy, c) ≈ -cond.gy
        @test K_s_ab(2, 1, cond.gx, cond.gy, c) ≈ -cond.gy
        @test K_s_ab(2, 3, cond.gx, cond.gy, c) ≈ -cond.gy
        @test K_s_ab(3, 2, cond.gx, cond.gy, c) ≈ -cond.gy
        # 1 и 3 не соседи: расстояние 2, но Nx=1, поэтому не подходит
        @test K_s_ab(1, 3, cond.gx, cond.gy, c) == 0.0
        @test K_s_ab(3, 1, cond.gx, cond.gy, c) == 0.0

        # все ячейки на i=1=Nx → gxa; только a=3 на k=Ny → gya
        @test K_a_ab(1, 1, cond.gxa, cond.gya, c) ≈ cond.gxa
        @test K_a_ab(2, 2, cond.gxa, cond.gya, c) ≈ cond.gxa
        @test K_a_ab(3, 3, cond.gxa, cond.gya, c) ≈ cond.gxa + cond.gya
    end
end

@testset "LinearSystem_right_side_vector" begin
    p  = PhysicsParam(H = 0.5)
    c  = Counts(2, 2, 2)
    gb = GridBody(p, c)
    gc = GridChannel(p, c, 1.0)
    cond = Conductivity(p, gb)

    # q = 1, Ta = t, Tin = 2t — простые функции, чтобы выкладки были устными
    src = Source((x, y, t) -> 1.0, t -> t)
    ch  = FlowChannel(s -> 0.0, t -> 2t)

    f = compute_right_side_function(c, gb, p, ch, src, cond)

    @testset "structure" begin
        @test length(f) == c.Nz
        @test all(el -> el isa Function, f)
    end

    @testset "body cells at t = 0: only source" begin
        # Ta(0) = 0, значит envoronment не даёт вклада
        for a in 1:c.Ns
            @test f[a](0.0) ≈ gb.Vs
        end
    end

    @testset "body cells at t = 1: source + envoronment" begin
        gxa = cond.gxa
        gya = cond.gya
        # индексы: a=1 (1,1), a=2 (2,1), a=3 (1,2), a=4 (2,2)
        expected_ga = [
            gxa,          # a=1: левая грань
            gxa,          # a=2: правая грань
            gxa + gya,    # a=3: левая + верхняя
            gxa + gya,    # a=4: правая + верхняя
        ]
        for a in 1:c.Ns
            @test f[a](1.0) ≈ gb.Vs + expected_ga[a]
        end
    end

    @testset "body cells without envoronment term" begin
        c3    = Counts(3, 3, 2)
        gb3   = GridBody(p, c3)
        cond3 = Conductivity(p, gb3)
        f3    = compute_right_side_function(c3, gb3, p, ch, src, cond3)

        # (i=2, k=1): нижний ряд, но не на боках и не сверху → нет Ta
        a2 = body_index(2, 1, c3)
        @test f3[a2](1.0) ≈ gb3.Vs

        # (i=2, k=2): внутренняя ячейка → нет Ta
        a5 = body_index(2, 2, c3)
        @test f3[a5](1.0) ≈ gb3.Vs

        # (i=3, k=2): правая грань, не верх → только gxa
        a6 = body_index(3, 2, c3)
        @test f3[a6](1.0) ≈ gb3.Vs + cond3.gxa

        # (i=2, k=3): верхняя грань → только gya
        a8 = body_index(2, 3, c3)
        @test f3[a8](1.0) ≈ gb3.Vs + cond3.gya
    end

    @testset "first channel cell: β * Tin(t)" begin
        @test f[c.Ns + 1](0.0) ≈ 0.0
        @test f[c.Ns + 1](1.0) ≈ p.β * 2.0
        @test f[c.Ns + 1](2.5) ≈ p.β * 5.0
    end

    @testset "remaining channel cells: zero for all t" begin
        for a in (c.Ns + 2):c.Nz
            @test f[a](0.0) == 0.0
            @test f[a](1.0) == 0.0
            @test f[a](-7.3) == 0.0
        end
    end

    @testset "body cells are linear in t" begin
        # f[a](t) = Vs + ga * t
        for a in 1:c.Ns
            for t in (0.0, 0.5, 1.0, 2.0)
                expected = gb.Vs + (f[a](1.0) - gb.Vs) * t
                @test f[a](t) ≈ expected
            end
        end
    end

    @testset "source time dependence" begin
        # q зависит от t, Ta = 0 — тогда envoronment выключен
        src_t = Source((x, y, t) -> t, t -> 0.0)
        f_t = compute_right_side_function(c, gb, p, ch, src_t, cond)
        for a in 1:c.Ns
            @test f_t[a](0.0) ≈ 0.0
            @test f_t[a](1.0) ≈ gb.Vs
            @test f_t[a](2.5) ≈ 2.5 * gb.Vs
        end
        # первая ячейка канала по-прежнему получает β * Tin(t)
        @test f_t[c.Ns + 1](1.0) ≈ p.β * 2.0
        @test f_t[c.Ns + 2](1.0) == 0.0
    end
end

@testset "K_matvec_par vs K_matvec" begin

    @testset "correctness" begin
        grids = [
            (PhysicsParam(H = 0.5), Counts(2,   2,   2)),
            (PhysicsParam(H = 0.5), Counts(5,   4,   3)),
            (PhysicsParam(H = 0.5), Counts(20,  6,  17)),
            (PhysicsParam(H = 0.5), Counts(120, 36, 102)),
        ]

        for (p, c) in grids
            gb  = GridBody(p, c)
            gc  = GridChannel(p, c)
            bnd = Boundary(gb, gc,
                           [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)],
                           [1.0, 2.0, 3.0])

            vectors = [
                ones(c.Nz),
                collect(1.0:c.Nz),
                [(-1.0)^k * k for k in 1:c.Nz],
                fill(-0.5, c.Nz),
                zeros(c.Nz),
            ]

            for v in vectors
                seq = K_matvec(v, c, bnd, p, gb)
                par = K_matvec_par(v, c, bnd, p, gb)
                @test length(par) == c.Nz
                @test par ≈ seq rtol=1e-12
            end
        end
    end

    @testset "linearity" begin
        p  = PhysicsParam(H = 0.5)
        c  = Counts(20, 6, 17)
        gb = GridBody(p, c)
        gc = GridChannel(p, c)
        bnd = Boundary(gb, gc,
                       [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)],
                       [1.0, 2.0, 3.0])

        v = collect(1.0:c.Nz)
        w = fill(2.0, c.Nz)
        @test K_matvec_par(v .+ w, c, bnd, p, gb) ≈
              K_matvec_par(v, c, bnd, p, gb) .+
              K_matvec_par(w, c, bnd, p, gb) rtol=1e-12
    end
end

@testset "K_matvec_par scaling" begin
    nt = Threads.nthreads()
    @info "Threads.nthreads() = $nt"

    sizes = [(200, 60, 170), (400, 120, 340), (800, 240, 680),
             (1200, 360, 1020), (1600, 480, 1360), (2400, 720, 2040)]

    for (Nx, Ny, Nc) in sizes
        p  = PhysicsParam(H = 0.5)
        c  = Counts(Nx, Ny, Nc)
        gb = GridBody(p, c)
        gc = GridChannel(p, c)
        bnd = Boundary(gb, gc,
                       [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)],
                       [1.0, 2.0, 3.0])

        v = rand(c.Nz)

        # прогрев
        K_matvec(v, c, bnd, p, gb)
        K_matvec_par(v, c, bnd, p, gb)

        t_seq = minimum(@elapsed K_matvec(v, c, bnd, p, gb) for _ in 1:20)
        t_par = minimum(@elapsed K_matvec_par(v, c, bnd, p, gb) for _ in 1:20)

        @info "grid $(Nx)x$(Ny)+$(Nc)" t_seq_ms = t_seq * 1e3 t_par_ms = t_par * 1e3 speedup = t_seq / t_par
    end
end

@testset "compute_right_side_function_par vs compute_right_side_function" begin

    @testset "correctness" begin
        grids = [
            (PhysicsParam(H = 0.5), Counts(2,   2,   2)),
            (PhysicsParam(H = 0.5), Counts(5,   4,   3)),
            (PhysicsParam(H = 0.5), Counts(20,  6,  17)),
            (PhysicsParam(H = 0.5), Counts(120, 36, 102)),
        ]

        times = [0.0, 0.5, 1.0, 2.5]

        for (p, c) in grids
            gb  = GridBody(p, c)
            gc  = GridChannel(p, c)
            cond = Conductivity(p, gb)
            src = Source((x, y, t) -> x + 10y + 100t, t -> 2t)
            ch  = FlowChannel(s -> 0.0, t -> 3t)

            f_seq = compute_right_side_function(c, gb, p, ch, src, cond)
            f_par = compute_right_side_function_par(c, gb, p, ch, src, cond)

            @test length(f_par) == c.Nz
            @test all(el -> el isa Function, f_par)

            for t in times
                for a in 1:c.Nz
                    @test f_par[a](t) ≈ f_seq[a](t) rtol=1e-12
                end
            end
        end
    end

    @testset "linearity in t" begin
        p  = PhysicsParam(H = 0.5)
        c  = Counts(20, 6, 17)
        gb = GridBody(p, c)
        cond = Conductivity(p, gb)
        src = Source((x, y, t) -> x + 10y + 100t, t -> 2t)
        ch  = FlowChannel(s -> 0.0, t -> 3t)

        f = compute_right_side_function_par(c, gb, p, ch, src, cond)

        for a in 1:c.Ns
            f0 = f[a](0.0)
            f1 = f[a](1.0)
            for t in (0.5, 1.5, 2.0)
                @test f[a](t) ≈ f0 + (f1 - f0) * t rtol=1e-12
            end
        end
    end
end

@testset "compute_right_side_function_par scaling" begin
    nt = Threads.nthreads()
    @info "Threads.nthreads() = $nt"

    sizes = [(200, 60, 170), (400, 120, 340), (800, 240, 680),
             (1200, 360, 1020), (1600, 480, 1360), (2400, 720, 2040)]

    for (Nx, Ny, Nc) in sizes
        p  = PhysicsParam(H = 0.5)
        c  = Counts(Nx, Ny, Nc)
        gb = GridBody(p, c)
        cond = Conductivity(p, gb)
        src = Source((x, y, t) -> x + 10y + 100t, t -> 2t)
        ch  = FlowChannel(s -> 0.0, t -> 3t)

        # прогрев
        compute_right_side_function(c, gb, p, ch, src, cond)
        compute_right_side_function_par(c, gb, p, ch, src, cond)

        t_seq = minimum(@elapsed compute_right_side_function(c, gb, p, ch, src, cond) for _ in 1:20)
        t_par = minimum(@elapsed compute_right_side_function_par(c, gb, p, ch, src, cond) for _ in 1:20)

        @info "grid $(Nx)x$(Ny)+$(Nc)" t_seq_ms = t_seq * 1e3 t_par_ms = t_par * 1e3 speedup = t_seq / t_par
    end
end

@testset "M_matvec_par vs M_matvec" begin

    @testset "correctness" begin
        grids = [
            (PhysicsParam(H = 0.5), Counts(2,   2,   2)),
            (PhysicsParam(H = 0.5), Counts(5,   4,   3)),
            (PhysicsParam(H = 0.5), Counts(20,  6,  17)),
            (PhysicsParam(H = 0.5), Counts(120, 36, 102)),
        ]

        for (p, c) in grids
            gb = GridBody(p, c)
            gc = GridChannel(p, c)

            vectors = [
                ones(c.Nz),
                collect(1.0:c.Nz),
                [(-1.0)^k * k for k in 1:c.Nz],
                fill(-0.5, c.Nz),
                zeros(c.Nz),
            ]

            for v in vectors
                seq = M_matvec(v, gb.Vs, p, gc)
                par = M_matvec_par(v, gb.Vs, p, gc)
                @test length(par) == c.Nz
                @test par ≈ seq rtol=1e-12
            end
        end
    end

    @testset "linearity" begin
        p  = PhysicsParam(H = 0.5)
        c  = Counts(20, 6, 17)
        gb = GridBody(p, c)
        gc = GridChannel(p, c)

        v = collect(1.0:c.Nz)
        w = fill(2.0, c.Nz)
        @test M_matvec_par(v .+ w, gb.Vs, p, gc) ≈
              M_matvec_par(v, gb.Vs, p, gc) .+
              M_matvec_par(w, gb.Vs, p, gc) rtol=1e-12
    end
end

@testset "M_matvec_par scaling" begin
    nt = Threads.nthreads()
    @info "Threads.nthreads() = $nt"

    sizes = [(200, 60, 170), (400, 120, 340), (800, 240, 680),
             (1200, 360, 1020), (1600, 480, 1360), (2400, 720, 2040)]

    for (Nx, Ny, Nc) in sizes
        p  = PhysicsParam(H = 0.5)
        c  = Counts(Nx, Ny, Nc)
        gb = GridBody(p, c)
        gc = GridChannel(p, c)

        v = rand(c.Nz)

        M_matvec(v, gb.Vs, p, gc)
        M_matvec_par(v, gb.Vs, p, gc)

        t_seq = minimum(@elapsed M_matvec(v, gb.Vs, p, gc) for _ in 1:20)
        t_par = minimum(@elapsed M_matvec_par(v, gb.Vs, p, gc) for _ in 1:20)

        @info "grid $(Nx)x$(Ny)+$(Nc)" t_seq_ms = t_seq * 1e3 t_par_ms = t_par * 1e3 speedup = t_seq / t_par
    end
end