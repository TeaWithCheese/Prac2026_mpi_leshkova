module MPIBuild
using ..Definitions
using ..LinearSystem

export local_M_matvec, local_K_matvec, local_f, local_all

function local_M_matvec(vec::Vector{Float64}, lo::Int, hi::Int,
                        Vs::Float64, params::PhysicsParam, gc::GridChannel, c::Counts)
    n = hi - lo + 1
    res = Vector{Float64}(undef, n)
    c_cA_c = params.c_c * params.A_c
    c_sVs = params.c_s * Vs
    for a in lo:hi
        if a <= c.Ns
            res[a - lo + 1] = c_sVs * vec[a]
        else
            j = a - c.Ns
            res[a - lo + 1] = c_cA_c * gc.ds[j] * vec[a]
        end
    end
    return res
end

function local_K_matvec(vec::Vector{Float64}, lo::Int, hi::Int,
                        c::Counts, boundary::Boundary, params::PhysicsParam, body_grid::GridBody)
    cond = LinearSystem.Conductivity(params, body_grid)
    n = hi - lo + 1
    res = zeros(Float64, n)
    Ns = c.Ns
    
    for a in max(lo, 1):min(hi, Ns)
        (i, k) = LinearSystem.inverse_body_index(a, c)
        val = 0.0
        if i < c.Nx; val += cond.gx * (vec[a] - vec[a + 1]); end
        if i > 1;    val += cond.gx * (vec[a] - vec[a - 1]); end
        if k < c.Ny; val += cond.gy * (vec[a] - vec[a + c.Nx]); end
        if k > 1;    val += cond.gy * (vec[a] - vec[a - c.Nx]); end
        
        ga = (i == 1 || i == c.Nx) * cond.gxa + (k == c.Ny) * cond.gya
        val += ga * vec[a]
        res[a - lo + 1] += val
    end
    
    for a in max(lo, Ns + 1):min(hi, c.Nz)
        j = a - Ns
        val = params.β * vec[a]
        if j > 1
            val -= params.β * vec[a - 1]
        end
        res[a - lo + 1] += val
    end
    
    for ((i, j, m), ℓ) in boundary.ell
        g = params.b * boundary.h[m] * ℓ
        a = LinearSystem.body_index(i, 1, c)
        b = LinearSystem.channel_index(j, c)
        d = g * (vec[a] - vec[b])
        
        if lo <= a <= hi
            res[a - lo + 1] += d
        end
        if lo <= b <= hi
            res[b - lo + 1] -= d
        end
    end
    return res
end

function local_f(t::Float64, lo::Int, hi::Int,
                 c::Counts, body_grid::GridBody, params::PhysicsParam,
                 ch::FlowChannel, src::Source, conductivity::LinearSystem.Conductivity)
    n = hi - lo + 1
    res = zeros(Float64, n)
    Vs = body_grid.Vs
    for a in lo:hi
        if a <= c.Ns
            (i, k) = LinearSystem.inverse_body_index(a, c)
            x = body_grid.xc[i]
            y = body_grid.yc[k]
            ga_x = conductivity.gxa * ((i == 1) + (i == c.Nx))
            ga_y = conductivity.gya * (k == c.Ny)
            res[a - lo + 1] = Vs * src.q(x, y, t) + (ga_x + ga_y) * src.Ta(t)
        elseif a == c.Ns + 1
            res[a - lo + 1] = params.β * ch.Tin(t)
        else
            res[a - lo + 1] = 0.0
        end
    end
    return res
end

function local_all(vec::Vector{Float64}, t::Float64, lo::Int, hi::Int,
                   c::Counts, boundary::Boundary, params::PhysicsParam,
                   body_grid::GridBody, gc::GridChannel, ch::FlowChannel, src::Source,
                   cond::LinearSystem.Conductivity)
    n = hi - lo + 1
    buf = Vector{Float64}(undef, 3 * n)
    
    Mres = local_M_matvec(vec, lo, hi, body_grid.Vs, params, gc, c)
    Kres = local_K_matvec(vec, lo, hi, c, boundary, params, body_grid)
    Fres = local_f(t, lo, hi, c, body_grid, params, ch, src, cond)
    
    @views buf[1:n]         .= Mres
    @views buf[n+1:2*n]     .= Kres
    @views buf[2*n+1:3*n]   .= Fres
    
    return buf
end

end