"""
    MPIBuild

Функции для построения полудискретной системы M dx/dt + Kx = f(t) рангами.
Каждый ранг локально вычисляет свою часть векторов (диапазон lo:hi).
Глобальная сборка выполняется на корневом процессе через MPI (Gatherv).
"""

module MPIBuild
using ..Definitions
using ..LinearSystem

export local_M_matvec, local_K_matvec, local_f, local_all


"""
    local_M_matvec(vec, lo, hi, Vs, params, gc, c)

Собирает lo::hi часть matvec с диагональной матрицей теплоёмкостей `M`.

# Аргументы
- `vec::Vector{Float64}` : Входной вектор температур (полной длины `c.Nz`).
- `lo::Int`              : Начальный глобальный индекс локального диапазона.
- `hi::Int`              : Конечный глобальный индекс локального диапазона.
- `Vs::Float64`          : Объём ячейки детали.
- `params::PhysicsParam` : Физические параметры (используются `c_s`, `c_c`, `A_c`).
- `gc::GridChannel`      : Сетка канала (используются длины ячеек `ds`).
- `c::Counts`            : Количества ячеек (используются `Ns`, `Nc`).

# Возвращает
- `Vector{Float64}` : Локальный кусок результата умножения на матрицу `M` длиной `hi - lo + 1`.
"""
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


"""
    local_K_matvec(vec, lo, hi, c, boundary, params, body_grid)

Вычисляет часть `lo:hi` произведения вектора температур на матрицу 
пространственных операторов `K` (теплопроводность, внешняя теплоотдача, 
контактный обмен и адвекция).

# Аргументы
- `vec::Vector{Float64}`     : Входной вектор температур (полной длины `c.Nz`).
- `lo::Int`                  : Начальный глобальный индекс локального диапазона.
- `hi::Int`                  : Конечный глобальный индекс локального диапазона.
- `c::Counts`                : Количества ячеек (используются `Ns`, `Nc`, `Nx`, `Ny`).
- `boundary::Boundary`       : Информация о контактных пересечениях (длины `ell`, коэффициенты `h`).
- `params::PhysicsParam`     : Физические параметры (используются `k_s`, `h_a`, `b`, `β`).
- `body_grid::GridBody`      : Сетка детали (используются шаги `dx`, `dy`).

# Возвращает
- `Vector{Float64}` : Локальный кусок результата умножения на матрицу `K` длиной `hi - lo + 1`.
"""
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

"""
    local_f(t, lo, hi, c, body_grid, params, ch, src, conductivity)

Вычисляет часть `lo:hi` вектора правой части `f(t)` полудискретной 
системы (источники тепла, внешняя среда и температура на входе).

# Аргументы
- `t::Float64`                      : Текущее время.
- `lo::Int`                         : Начальный глобальный индекс локального диапазона.
- `hi::Int`                         : Конечный глобальный индекс локального диапазона.
- `c::Counts`                       : Количества ячеек.
- `body_grid::GridBody`             : Сетка детали (используются объём `Vs`, координаты центров `xc`, `yc`).
- `params::PhysicsParam`            : Физические параметры (используется `β`).
- `ch::FlowChannel`                 : Параметры канала (используется функция `Tin`).
- `src::Source`                     : Источники тепла (используются функции `q` и `Ta`).
- `conductivity::Conductivity`      : Проводимости внешней теплоотдачи (используются `gxa`, `gya`).

# Возвращает
- `Vector{Float64}` : Локальный кусок вектора правой части `f(t)` длиной `hi - lo + 1`.
"""
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


"""
    local_all(vec, t, lo, hi, c, boundary, params, body_grid, gc, ch, src, cond)

Вычисляет локальные части `lo:hi` для операторов `M`, `K` и правой части `f(t)`,
объединяя их в единый буфер для последующей MPI-сборки (Gatherv).

# Аргументы
- `vec::Vector{Float64}`             : Входной вектор температур.
- `t::Float64`                       : Текущее время.
- `lo::Int`, `hi::Int`               : Глобальные индексы локального диапазона.
- `c::Counts`                        : Количества ячеек.
- `boundary::Boundary`               : Контактные пересечения.
- `params::PhysicsParam`             : Физические параметры.
- `body_grid::GridBody`              : Сетка детали.
- `gc::GridChannel`                  : Сетка канала.
- `ch::FlowChannel`                  : Параметры потока (граничные условия).
- `src::Source`                      : Источники тепла.
- `cond::Conductivity`               : Проводимости.

# Возвращает
- `Vector{Float64}` : Буфер длины `3 * (hi - lo + 1)`, содержащий `[M*vec, K*vec, f(t)]`.
"""
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