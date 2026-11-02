"""
    Definitions

Общие определения модели тепловой диагностики: физические параметры,
сетки детали и канала, структуры источника, детали и канала.

Модуль описывает типы данных,
используемые всеми этапами проекта.
"""
module Definitions

export PhysicsParam, Counts, GridBody, GridChannel,
       Source, Body, FlowChannel, Boundary, get_ell

"""
    PhysicsParam

Физические параметры модели, заданные на всех этапах

# Поля
- `L::Float64` : длина детали и канала, м.
- `H::Float64` : высота детали, м.
- `b::Float64` : толщина детали, м.
- `c_s::Float64` : объёмная теплоёмкость детали, Дж/(м³·К).
- `k_s::Float64` : теплопроводность детали, Вт/(м·К).
- `h_a::Float64` : коэффициент внешней теплоотдачи на трёх сторонах детали, Вт/(м²·К).
- `A_c::Float64` : площадь поперечного сечения канала, м².
- `c_c::Float64` : объёмная теплоёмкость жидкости, Дж/(м³·К).
- `β::Float64` : тепловой эквивалент расхода `c_c * Vdot`, Вт/К.
"""
struct PhysicsParam
    L::Float64
    H::Float64
    b::Float64
    c_s::Float64
    k_s::Float64
    h_a::Float64
    A_c::Float64
    c_c::Float64
    β::Float64
end


"""
    PhysicsParam(; L=1.0, H=0.3, b=1.0, c_s=1.0, k_s=0.1, h_a=0.03,
                   A_c=0.05, c_c=1.0, β=0.1)

Конструктор с значениями по умолчанию из начальных данных первого этапа.
Все параметры безразмерные.
"""
PhysicsParam(; L=1.0, H=0.3, b=1.0,
               c_s=1.0, k_s=0.1, h_a=0.03,
               A_c=0.05, c_c=1.0, β=0.1) =
    PhysicsParam(L, H, b, c_s, k_s, h_a, A_c, c_c, β)

"""
    Counts

Числа ячеек сеток детали и канала.

# Поля
- `Nx::Int` : число ячеек детали по x.
- `Ny::Int` : число ячеек детали по y.
- `Ns::Int` : `Nx * Ny`, число ячеек детали.
- `Nc::Int` : число ячеек канала.
- `Nz::Int` : `Ns + Nc`, полное число температур.
"""
struct Counts
    Nx::Int
    Ny::Int
    Ns::Int
    Nc::Int
    Nz::Int
end

Counts(Nx::Int, Ny::Int, Nc::Int) = Counts(Nx, Ny, Nx * Ny, Nc, Nx * Ny + Nc)


"""
    GridBody

Прямоугольная равномерная сетка детали.

# Поля
- `Nx::Int` : число ячеек по x.
- `Ny::Int` : число ячеек по y.
- `Ns::Int` : `Nx * Ny`, полное число ячеек.
- `dx::Float64` : шаг сетки по x, `dx = L / Nx`.
- `dy::Float64` : шаг сетки по y, `dy = H / Ny`.
- `xc::Vector{Float64}` : центры ячеек по x, `xc[i] = (i - 1/2) dx`,
  `i = 1..Nx`, длина `Nx`.
- `yc::Vector{Float64}` : центры ячеек по y, `yc[k] = (k - 1/2) dy`,
  `k = 1..Ny`, длина `Ny`.
- `Vs::Float64` : объём ячейки `b * dx * dy`.

Нумерация: узел `(i, k)` соответствует одномерному индексу
`E = i + Nx * (k - 1)`, `i = 1..Nx`, `k = 1..Ny`.
"""
struct GridBody
    Nx::Int
    Ny::Int
    Ns::Int
    dx::Float64
    dy::Float64
    xc::Vector{Float64}
    yc::Vector{Float64}
    Vs::Float64
end


"""
    GridBody(p::PhysicsParam, c::Counts) -> GridBody

Строит равномерную сетку детали.

Шаги: `dx = p.L / c.Nx`, `dy = p.H / c.Ny`.
Центры: `xc[i] = (i - 1/2) dx`, `yc[k] = (k - 1/2) dy`.
Объём ячейки: `Vs = p.b * dx * dy`.
"""
function GridBody(p::PhysicsParam, c::Counts)
    dx = p.L / c.Nx
    dy = p.H / c.Ny
    xc = [(i - 0.5) * dx for i in 1:c.Nx]
    yc = [(k - 0.5) * dy for k in 1:c.Ny]
    Vs = p.b * dx * dy
    return GridBody(c.Nx, c.Ny, c.Ns, dx, dy, xc, yc, Vs)
end

"""
    GridChannel

Одномерная неравномерная сетка канала.

# Поля
- `Nc::Int` : число ячеек.
- `sb::Vector{Float64}` : границы ячеек `sb[j] = s_{j-1/2}`, длина `Nc + 1`,
  `sb[1] = 0`, `sb[Nc + 1] = L`.
- `sc::Vector{Float64}` : центры ячеек `sc[j] = (sb[j] + sb[j+1]) / 2`,
  `j = 1..Nc`, длина `Nc`.
- `ds::Vector{Float64}` : длины ячеек `ds[j] = sb[j+1] - sb[j]`,
  `j = 1..Nc`, длина `Nc`.
"""
struct GridChannel
    Nc::Int
    sb::Vector{Float64}
    sc::Vector{Float64}
    ds::Vector{Float64}
end

"""
    GridChannel(p::PhysicsParam, c::Counts, power::Float64 = 1.15) -> GridChannel

Строит неравномерную сетку канала.

Границы: `sb[j+1] = p.L * (j / c.Nc)^power`, `j = 0..c.Nc`.
Центры: `sc[j] = (sb[j] + sb[j+1]) / 2`.
Длины: `ds[j] = sb[j+1] - sb[j]`.
"""
function GridChannel(p::PhysicsParam, c::Counts, power::Float64 = 1.15)
    sb = [p.L * (j / c.Nc)^power for j in 0:c.Nc]
    sc = [(sb[j] + sb[j + 1]) / 2 for j in 1:c.Nc]
    ds = [sb[j + 1] - sb[j] for j in 1:c.Nc]
    return GridChannel(c.Nc, sb, sc, ds)
end


"""
    Source

Источник тепла внутри детали и температура внешней среды.

# Поля
- `q::Function`  : плотность источника `q(x, y, t) -> Float64`, Вт/м³.
- `Ta::Function` : температура внешней среды на трёх сторонах детали,
  `Ta(t) -> Float64`, К.
"""
struct Source
    q::Function
    Ta::Function
end


"""
    Body

Деталь. Уникальные параметры детали, не связанные с дискретизацией.

# Поля
- `Ts0::Function` : начальная температура `Ts0(x, y) -> Float64`.
"""
struct Body
    Ts0::Function
end


"""
    FlowChannel

Канал. Уникальные параметры канала, не связанные с дискретизацией.

# Поля
- `Tc0::Function` : начальная температура `Tc0(s) -> Float64`.
- `Tin::Function` : температура на входе `Tin(t) -> Float64`.
"""
struct FlowChannel
    Tc0::Function
    Tin::Function
end

"""
    Boundary

Контакт между деталью и каналом — поверхность `y = 0`.
Разбит на `M` отрезков `K_m = (xL_m, xR_m)` с постоянным
коэффициентом контакта `h_m` на каждом отрезке.

# Поля
- `M::Int` : число контактных отрезков.
- `segments::Vector{Tuple{Float64,Float64}}` : границы отрезков
  `K_m = (xL_m, xR_m)`, `m = 1..M`, вдоль оси x.
- `h::Vector{Float64}` : коэффициенты `h_m`, Вт/(м²·К), `m = 1..M`.
  `h_m` — постоянное на отрезке `K_m` значение.
  В 1 этапе заданы, во 2 этапе искомые.
- `ell::Dict{Tuple{Int,Int,Int},Float64}`: предвычисленные длины
  пересечений `ℓ[i,j,m]` трёх множеств — ячейка детали `I_i`,
  ячейка канала `[s_{j-1/2}, s_{j+1/2}]`, отрезок `K_m`.
  Хранятся только ненулевые тройки `(i, j, m, ℓ)`.
"""
struct Boundary
    M::Int
    segments::Vector{Tuple{Float64,Float64}}
    h::Vector{Float64}
    ell::Dict{Tuple{Int,Int,Int},Float64}
end

"""
    Boundary(gb::GridBody, gc::GridChannel,
             segments::Vector{Tuple{Float64,Float64}},
             h::Vector{Float64}) -> Boundary

Строит контакт между деталью и каналом на поверхности `y = 0`.

`segments[m] = (xL_m, xR_m)` — произвольные отрезки вдоль оси x,
`h[m]` — постоянный коэффициент контакта на `K_m`.
Отрезки не пересекаются и не имеют зазоров.

Для каждой тройки (ячейка детали `I_i = [(i-1)Δx, i Δx]`,
ячейка канала `[sb[j], sb[j+1]]`, отрезок `K_m`) вычисляется длина
пересечения

    ℓ_{ijm} = max(0, min(i Δx, sb[j+1], xR_m)
                   - max((i-1) Δx, sb[j], xL_m)),

и в `ell` сохраняются только тройки с `ℓ > 0`.

# Ошибки
- `ArgumentError`, если длины `segments` и `h` не совпадают.
- `ArgumentError`, если `h` пуст или содержит неположительные значения.
- `ArgumentError`, если сумма длин пересечений для некоторого `K_m`
  не совпадает с длиной отрезка.
"""
function Boundary(gb::GridBody, gc::GridChannel,
                  segments::Vector{Tuple{Float64,Float64}},
                  h::Vector{Float64})
    M = length(segments)
    M > 0 || throw(ArgumentError("segments должен быть непустым"))
    length(h) == M || throw(ArgumentError("длины segments и h не совпадают"))
    all(h .> 0) || throw(ArgumentError("все h_m должны быть положительны"))

    results = Vector{Vector{Tuple{Int,Int,Float64}}}(undef, M)

    tasks = map(1:M) do m
        Threads.@spawn begin
            xL, xR = segments[m]
            local_vec = Tuple{Int,Int,Float64}[]
            total = 0.0
            for i in 1:gb.Nx
                iL = (i - 1) * gb.dx
                iR = i * gb.dx
                for j in 1:gc.Nc
                    ℓ = min(iR, gc.sb[j+1], xR) - max(iL, gc.sb[j], xL)
                    if ℓ > 0
                        push!(local_vec, (i, j, ℓ))
                        total += ℓ
                    end
                end
            end
            isapprox(total, xR - xL; atol = 1e-12) ||
                throw(ArgumentError("сумма пересечений отрезка K_$m равна $total, ожидалось $(xR - xL)"))
            results[m] = local_vec
        end
    end
    
    for t in tasks
        try
            fetch(t)
        catch e
            if e isa TaskFailedException
                throw(e.task.exception)
            else
                rethrow()
            end
        end
    end
    
    ell = Dict{Tuple{Int,Int,Int},Float64}()
    for m in 1:M
        for (i, j, ℓ) in results[m]
            ell[(i, j, m)] = ℓ
        end
    end

    return Boundary(M, segments, h, ell)
end

"""
    get_ell(boundary, i, j, m) -> Float64

Длина пересечения `ℓ_{ijm}` для тройки `(i, j, m)`, или `0.0`,
если тройка отсутствует (пересечение пустое).
"""
function get_ell(boundary::Boundary, i::Int, j::Int, m::Int)
    return get(boundary.ell, (i, j, m), 0.0)
end

end