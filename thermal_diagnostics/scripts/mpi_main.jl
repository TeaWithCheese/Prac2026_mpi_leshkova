using MPI
using thermal_diagnostics
using thermal_diagnostics.Definitions
using thermal_diagnostics.LinearSystem
using thermal_diagnostics.MPIBuild
using Random
using Printf

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Параметры запускаемой задачи
    p = PhysicsParam(H = 0.5)
    Nx, Ny, Nc = 2400, 720, 2040
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

    function run_once!()
        if n > 0
            M_loc = local_M_matvec(vec, lo, hi, gb.Vs, p, gc, c)
            K_loc = local_K_matvec(vec, lo, hi, c, bnd, p, gb)
            f_loc = local_f(t, lo, hi, c, gb, p, ch, src, cond)
            send_buf[1:n] .= M_loc
            send_buf[n+1:2*n] .= K_loc
            send_buf[2*n+1:3*n] .= f_loc
        end
        MPI.Gatherv!(send_buf, recv_vbuf, 0, comm)
    end

    # Прогрев
    run_once!()

    # Замер
    iterations = 10
    times = Vector{Float64}(undef, iterations)
    for i in 1:iterations
        MPI.Barrier(comm)
        t_start = time_ns()
        run_once!()
        t_end = time_ns()
        times[i] = (t_end - t_start) / 1e9
    end

    min_time = minimum(times)
    all_times = MPI.Gather(min_time, 0, comm)

    if rank == 0
        max_time = maximum(all_times)
        @printf("MPI p=%d, Grid=%dx%d+%d, Time (max over ranks) = %.6f sec\n",
                nprocs, Nx, Ny, Nc, max_time)
        open("scaling_results.csv", "a") do io
            @printf(io, "%d,%.6f\n", nprocs, max_time)
        end
    end

    MPI.Finalize()
end

main()