using MPI
using thermal_diagnostics
using thermal_diagnostics.Definitions
using thermal_diagnostics.LinearSystem
using thermal_diagnostics.MPIBuild
using Random
using Printf

# Тестовая программа для проверки на локальной машине логики mpi_main

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    p = PhysicsParam(H = 0.5)
    Nx, Ny, Nc = 200, 60, 170
    c = Counts(Nx, Ny, Nc)
    gb = GridBody(p, c)
    gc = GridChannel(p, c)
    bnd = Boundary(gb, gc, [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)], [1.0, 2.0, 3.0])
    cond = Conductivity(p, gb)
    src = Source((x, y, t) -> x + 10y + 100t, t -> 2t)
    ch = FlowChannel(s -> 0.0, t -> 3t)

    vec = rank == 0 ? rand(c.Nz) : Vector{Float64}(undef, c.Nz)
    MPI.Bcast!(vec, 0, comm)
    t = 1.0

    q = div(c.Nz, nprocs)
    r = rem(c.Nz, nprocs)
    lo = rank * q + min(rank, r) + 1
    hi = (rank + 1) * q + min(rank + 1, r)
    n = max(0, hi - lo + 1)

    send_buf = Vector{Float64}(undef, 3 * n)
    recv_buf = rank == 0 ? Vector{Float64}(undef, 3 * c.Nz) : nothing

    lengths = zeros(Int32, nprocs)
    for i in 0:nprocs-1
        lo_i = i * q + min(i, r) + 1
        hi_i = (i + 1) * q + min(i + 1, r)
        lengths[i + 1] = Int32(max(0, hi_i - lo_i + 1))
    end
    counts = 3 .* lengths
    recv_vbuf = rank == 0 ? VBuffer(recv_buf, counts) : VBuffer(nothing)

    if n > 0
        M_loc = local_M_matvec(vec, lo, hi, gb.Vs, p, gc, c)
        K_loc = local_K_matvec(vec, lo, hi, c, bnd, p, gb)
        f_loc = local_f(t, lo, hi, c, gb, p, ch, src, cond)
        send_buf[1:n] .= M_loc
        send_buf[n+1:2*n] .= K_loc
        send_buf[2*n+1:3*n] .= f_loc
    end
    MPI.Gatherv!(send_buf, recv_vbuf, 0, comm)

    if rank == 0
        M_mpi = Vector{Float64}(undef, c.Nz)
        K_mpi = Vector{Float64}(undef, c.Nz)
        f_mpi = Vector{Float64}(undef, c.Nz)
        idx = 1
        for i in 1:nprocs
            n_i = lengths[i]
            if n_i > 0
                M_mpi[idx : idx + n_i - 1] .= recv_buf[3*(idx-1) + 1 : 3*(idx-1) + n_i]
                K_mpi[idx : idx + n_i - 1] .= recv_buf[3*(idx-1) + n_i + 1 : 3*(idx-1) + 2*n_i]
                f_mpi[idx : idx + n_i - 1] .= recv_buf[3*(idx-1) + 2*n_i + 1 : 3*(idx-1) + 3*n_i]
            end
            idx += n_i
        end

        M_seq = M_matvec(vec, gb.Vs, p, gc)
        K_seq = K_matvec(vec, c, bnd, p, gb)
        f_seq = compute_right_side_function(c, gb, p, ch, src, cond)
        f_val = [fn(t) for fn in f_seq]

        err_M = maximum(abs.(M_mpi .- M_seq))
        err_K = maximum(abs.(K_mpi .- K_seq))
        err_f = maximum(abs.(f_mpi .- f_val))

        @printf("TEST p=%d: Errors max|M| = %.2e, max|K| = %.2e, max|f| = %.2e\n",
                nprocs, err_M, err_K, err_f)
        (err_M < 1e-12 && err_K < 1e-12 && err_f < 1e-12) ? println("PASS") : println("FAIL")
    end

    MPI.Finalize()
end

main()