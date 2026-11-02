"""
    LinearSystem

Функции для дискретизации непрерывных уравнений по пространству и построения линейный системы

Линейная система имеет вид Mx' + Kx = f(t)
"""

module LinearSystem
using ..Definitions

export M_matvec, K_matvec, inverse_body_index, inverse_channel_index,
    g_x, g_y, g_x_a, g_y_a, Conductivity, K_ab, K_s_ab, K_c_ab, K_Г_ab, K_a_ab,
    g_ijm, body_index, channel_index, compute_right_side_function, 
    K_matvec_par, compute_right_side_function_par, M_matvec_par

"""
    ismod(x, y, z) -> Bool

Проверяет, сравнимы ли `x` и `y` по модулю `z`, то есть
`x ≡ y (mod z)`.

    ismod(x, y, z) = (x % z == y % z)

Аргументы `x`, `y` неотрицательные,

# Аргументы
- `x`, `y` : неотрицательные целые числа.
- `z`      : положительный целый модуль.

# Ошибки
- `ArgumentError`, если `z ≤ 0`.
"""
function ismod(x::Int, y::Int, z::Int)
    z > 0 || throw(ArgumentError("модуль z должен быть положительным, получено $z"))
    return x % z == y % z
end

"""
    body_index

Сопоставляет ячейке детали `(i, k)` сквозной индекс в векторе `x`

"""
function body_index(i::Int, k::Int, c::Counts)
    return (i + c.Nx * (k - 1))
end

"""
    inverse_body_index(a, c) -> (Int, Int)

Восстанавливает пару `(i, k)` ячейки детали по её сквозному индексу `a`
в векторе температур `x`.

Обратное отображение к `a = i + Nx * (k - 1)`:

    i = ((a - 1) % Nx) + 1
    k = div(a - 1, Nx) + 1

Здесь `i = 1..Nx` — номер ячейки по x, `k = 1..Ny` — номер по y.

# Аргументы
- `a` : сквозной индекс ячейки детали, `1 ≤ a ≤ Ns`.
- `c` : `Counts` с полем `Nx`.

# Возвращает
Кортеж `(i, k)`.
"""
function inverse_body_index(a::Int, c::Counts)
    i = ((a - 1) % c.Nx) + 1
    k = div(a - 1, c.Nx) + 1

    return (i, k)
end

"""
    channel_index

Сопоставляет ячейке канала `j` сквозной индекс в векторе `x`

"""
function channel_index(j::Int, c::Counts)
    return (c.Ns + j)
end

"""
    inverse_channel_index(a, c) -> Int

Восстанавливает номер ячейки канала `j` по её сквозному индексу `a`
в векторе температур `x`.

Обратное отображение к `a = Ns + j`:

    j = a - Ns

# Аргументы
- `a` : сквозной индекс ячейки канала, `Ns + 1 ≤ a ≤ Nz`.
- `c` : `Counts` с полем `Ns`.

# Возвращает
Номер ячейки канала `j`, `1 ≤ j ≤ Nc`.
"""
function inverse_channel_index(a::Int, c::Counts)
    j = a - c.Ns

    return j
end

"""
    g_x(k_s, b, dx, dy) -> Float64

Проводимость горизонтальной связи между соседними ячейками детали
(вдоль оси x).

    g_x = k_s * b * dy / dx

# Аргументы
- `k_s` : теплопроводность детали, Вт/(м·К).
- `b`   : толщина детали, м.
- `dx`  : шаг сетки по x, м.
- `dy`  : шаг сетки по y, м.
"""
function g_x(k_s::Float64, b::Float64, dx::Float64, dy::Float64)
    return k_s * b * dy / dx
end

"""
    g_x_a(h_a, b, dy) -> Float64

Проводимость внешней теплоотдачи через боковую грань детали
(левая при `i = 1` или правая при `i = Nx`).

    g_x_a = h_a * dy * b

# Аргументы
- `h_a` : коэффициент внешней теплоотдачи, Вт/(м²·К).
- `b`   : толщина детали, м.
- `dy`  : шаг сетки по y, м.
"""
function g_x_a(h_a::Float64, b::Float64, dy::Float64)
    return h_a * dy * b
end

"""
    g_y(k_s, b, dx, dy) -> Float64

Проводимость вертикальной связи между соседними ячейками детали
(вдоль оси y).

    g_y = k_s * b * dx / dy

# Аргументы
- `k_s` : теплопроводность детали, Вт/(м·К).
- `b`   : толщина детали, м.
- `dx`  : шаг сетки по x, м.
- `dy`  : шаг сетки по y, м.
"""
function g_y(k_s::Float64, b::Float64, dx::Float64, dy::Float64)
    return k_s * b * dx / dy
end

"""
    g_y_a(h_a, b, dx) -> Float64

Проводимость внешней теплоотдачи через верхнюю грань детали
(`k = Ny`).

    g_y_a = h_a * dx * b

# Аргументы
- `h_a` : коэффициент внешней теплоотдачи, Вт/(м²·К).
- `b`   : толщина детали, м.
- `dx`  : шаг сетки по x, м.
"""
function g_y_a(h_a::Float64, b::Float64, dx::Float64)
    return h_a * dx * b
end


"""
    Conductivity

Набор проводимостей детали, зависящих только от параметров задачи и
сетки, но не от индексов `(a, b)`. Вычисляется один раз и передаётся
во все функции, где нужны проводимости.

# Поля
- `gx::Float64`  : проводимость горизонтальной связи между соседними
  ячейками детали, `g_x(k_s, b, dx, dy) = k_s * b * dy / dx`.
- `gxa::Float64` : проводимость боковой грани детали (`i = 1` или
  `i = Nx`) для внешней теплоотдачи,
  `g_x_a(h_a, b, dy) = h_a * dy * b`.
- `gy::Float64`  : проводимость вертикальной связи между соседними
  ячейками детали, `g_y(k_s, b, dx, dy) = k_s * b * dx / dy`.
- `gya::Float64` : проводимость верхней грани детали (`k = Ny`)
  для внешней теплоотдачи,
  `g_y_a(h_a, b, dx) = h_a * dx * b`.
"""
struct Conductivity
    gx::Float64
    gxa::Float64
    gy::Float64
    gya::Float64
end

"""
    Conductivity(params::PhysicsParam, body_grid::GridBody) -> Conductivity

Строит `Conductivity` из физических параметров и сетки детали.

    gx  = g_x(params.k_s, params.b, body_grid.dx, body_grid.dy)
    gxa = g_x_a(params.h_a, params.b, body_grid.dy)
    gy  = g_y(params.k_s, params.b, body_grid.dx, body_grid.dy)
    gya = g_y_a(params.h_a, params.b, body_grid.dx)
"""
function Conductivity(params::PhysicsParam, body_grid::GridBody)
    gx  = g_x(params.k_s, params.b, body_grid.dx, body_grid.dy)
    gxa = g_x_a(params.h_a, params.b, body_grid.dy)
    gy  = g_y(params.k_s, params.b, body_grid.dx, body_grid.dy)
    gya = g_y_a(params.h_a, params.b, body_grid.dx)

    return Conductivity(gx, gxa, gy, gya)
end

"""
    M_matvec(vec, Vs, params, channel) -> Vector{Float64}

Умножает вектор температур `vec` на диагональную матрицу теплоёмкостей `M`.

Вектор имеет длину `Nz = Ns + Nc`, где `Ns` — число ячеек детали,
`Nc` — число ячеек канала. Первые `Ns` компонент — температуры детали,
последние `Nc` — температуры канала.

Результат:

- для ячеек детали (`1 ≤ a ≤ Ns`):
    `(M * vec)[a] = c_s * Vs * vec[a]`
- для ячеек канала (`a = Ns + j`, `j = 1..Nc`):
    `(M * vec)[Ns + j] = c_c * A_c * ds[j] * vec[Ns + j]`

# Аргументы
- `vec`     : вектор длины `Nz = Ns + Nc`.
- `Vs`      : объём ячейки детали, `Vs = b * dx * dy`.
- `params`  : `PhysicsParam`; используются поля `c_s`, `c_c`, `A_c`.
- `channel` : `GridChannel`; используются поля `Nc`, `ds`.

# Возвращает
Новый вектор `M * vec` длины `Nz`.
"""
function M_matvec(vec::Vector{Float64}, Vs::Float64, params::PhysicsParam, channel::GridChannel)
    Nz = length(vec)
    Nc = channel.Nc
    Ns = Nz - Nc
    res = Vector{Float64}(undef, Nz)

    c_cA_c = params.c_c * params.A_c
    c_sVs = params.c_s * Vs
    for ind in 1:Ns
        res[ind] = c_sVs * vec[ind]
    end

    for ind in 1:Nc
        res[Ns + ind] = c_cA_c * channel.ds[ind] * vec[Ns + ind]
    end

    return res
end

"""
    M_matvec_par(vec, Vs, params, channel) -> Vector{Float64}

Параллельный вариант `M_matvec`: умножение вектора температур на матрицу `M`.

# Аргументы
- `vec`     : вектор длины `Nz`.
- `Vs`      : объём ячейки детали.
- `params`  : `PhysicsParam`.
- `channel` : `GridChannel`.

# Возвращает
Новый вектор `M * vec` длины `Nz`.
"""
function M_matvec_par(vec::Vector{Float64}, Vs::Float64,
                      params::PhysicsParam, channel::GridChannel)
    Nz = length(vec)
    Nc = channel.Nc
    Ns = Nz - Nc
    res = Vector{Float64}(undef, Nz)

    c_cA_c = params.c_c * params.A_c
    c_sVs = params.c_s * Vs

    Threads.@threads for ind in 1:Ns
        res[ind] = c_sVs * vec[ind]
    end

    Threads.@threads for ind in 1:Nc
        res[Ns + ind] = c_cA_c * channel.ds[ind] * vec[Ns + ind]
    end

    return res
end

"""
    K_s_ab(a, b, g_x, g_y, c) -> Float64

Элемент `(a, b)` матрицы `K_s` — разностной диффузии детали.

Индексы `a`, `b` — сквозные номера ячеек: `a = i + Nx*(k-1)`, `i = 1..Nx`,
`k = 1..Ny`. Для ячеек канала (`a > Ns` или `b > Ns`) элемент равен нулю.

Диагональный элемент (`a == b`, `a ≤ Ns`) равен сумме проводимостей
всех имеющихся у ячейки `a` соседей:

    K_s_ab = g_x * ([a % Nx ≠ 0] + [a % Nx ≠ 1])
           + g_y * ([a > Nx] + [a ≤ Ns - Nx])

Недиагональный элемент (`a ≠ b`, `a, b ≤ Ns`) равен `-g` для каждой
реально соседней пары:

    K_s_ab = -g_x,  если |a - b| = 1 и min(a, b) % Nx ≠ 0
    K_s_ab = -g_y,  если |a - b| = Nx

иначе `K_s_ab = 0`.

# Аргументы
- `a`, `b`  : сквозные индексы ячеек, `1..Nz`.
- `g_x`     : проводимость горизонтальной связи, `g_x(k_s, b, dx, dy)`.
- `g_y`     : проводимость вертикальной связи, `g_y(k_s, b, dx, dy)`.
- `c`       : `Counts` с полями `Nx`, `Ns`.

# Возвращает
`Float64` — значение элемента `K_s[a, b]`.
"""
function K_s_ab(a::Int, b::Int, g_x::Float64, g_y::Float64, c::Counts)
    res = 0.0

    if (a == b) && (a ≤ c.Ns)
        if !ismod(a, 0, c.Nx)
            res += g_x
        end 

        if !ismod(a, 1, c.Nx)
            res += g_x
        end

        if (a > c.Nx)
            res += g_y
        end

        if (a ≤ c.Ns - c.Nx)
            res += g_y
        end
    elseif (a ≤ c.Ns) && (b ≤ c.Ns)
        lo = min(a, b)
        hi = max(a, b)
        if (hi - lo == 1) && !ismod(lo, 0, c.Nx)
            res -= g_x
        elseif hi - lo == c.Nx
            res -= g_y
        end
    end

    return res
end

"""
    K_a_ab(a, b, g_x_a, g_y_a, c) -> Float64

Элемент `(a, b)` матрицы `K_a` — внешней теплоотдачи с трёх сторон
детали (левой `i = 1`, правой `i = Nx`, верхней `k = Ny`).

Матрица диагональная:

    K_a_ab = g_x_a * ([i = 1] + [i = Nx]) + g_y_a * [k = Ny],  если a = b ≤ Ns,
    K_a_ab = 0,                                                 иначе.

Здесь `(i, k)` восстанавливаются из `a` через `inverse_body_index`.
Для угловой ячейки складываются оба
слагаемых — по числу граничных сторон.

# Аргументы
- `a`, `b`  : сквозные индексы, `1..Nz`.
- `g_x_a`   : проводимость боковой грани, `g_x_a(h_a, b, dy)`.
- `g_y_a`   : проводимость верхней грани, `g_y_a(h_a, b, dx)`.
- `c`       : `Counts` с полями `Nx`, `Ny`, `Ns`.

# Возвращает
`Float64` — значение элемента `K_a[a, b]`.
"""
function K_a_ab(a::Int, b::Int, g_x_a::Float64, g_y_a::Float64, c::Counts)
    res = 0.0

    if (a == b) && (a ≤ c.Ns)
        (i, k) = inverse_body_index(a, c)
        if (i == 1) || (i == c.Nx)
            res += g_x_a
        end 

        if k == c.Ny
            res += g_y_a
        end
    end

    return res
end

"""
    g_ijm(i, j, m, b, boundary) -> Float64

Контактная проводимость между нижней ячейкой детали `i` (при `k = 1`)
и ячейкой канала `j` на отрезке `K_m`.

    g_ijm = b * h[m] * ℓ_{ijm}

где `ℓ_{ijm}` — длина пересечения трёх отрезков: ячейки детали `I_i`,
ячейки канала `[sb[j], sb[j+1]]` и контактного отрезка `K_m`.
Берётся через `get_ell(boundary, i, j, m)`; если пересечения нет,
возвращается `0.0`.

# Аргументы
- `i`        : номер ячейки детали по x, `1..Nx`.
- `j`        : номер ячейки канала, `1..Nc`.
- `m`        : номер контактного отрезка, `1..M`.
- `b`        : толщина детали, м.
- `boundary` : `Boundary` с полями `h`, `ell`.

# Возвращает
`Float64` — значение `g_ijm`, Вт/К. Ноль, если пересечение пусто.
"""
function g_ijm(i::Int, j::Int, m::Int, b::Float64, boundary::Boundary)
    ℓ = get_ell(boundary, i, j, m)
    return b * boundary.h[m] * ℓ
end

"""
    K_Г_ab(a, b, c, b_body, boundary) -> Float64

Элемент `(a, b)` матрицы `K_Γ` — контактного обмена между нижним рядом
ячеек детали (`k = 1`) и ячейками канала.

Матрица симметрична, отлична от нуля только в блоках деталь—деталь,
канал—канал (диагональ) и деталь—канал (недиагональ):

- `a = b ≤ Ns`, ячейка детали с `k = 1`:
    `K_Г_ab = Σ_{j,m} g_ijm`,
  где `i` — номер по x, `g_ijm = b * h[m] * ℓ_{ijm}`.
- `a = b > Ns`, ячейка канала `j`:
    `K_Г_ab = Σ_{i,m} g_ijm`, сумма по всем нижним ячейкам детали `i`.
- одна из ячеек — деталь с `k = 1`, другая — канал:
    `K_Г_ab = -Σ_m g_ijm`, где `i` — ячейка детали, `j` — ячейка канала.
- все остальные случаи: `K_Г_ab = 0`.

Слагаемое берётся только для ячеек детали с `k = 1`; для остальных
рядов ячеек детали контакт отсутствует.

# Аргументы
- `a`, `b`    : сквозные индексы, `1..Nz`.
- `c`         : `Counts` с полями `Nx`, `Ns`, `Nc`.
- `b_body`    : толщина детали, м.
- `boundary`  : `Boundary` с полями `M`, `h`, `ell`.

# Возвращает
`Float64` — значение элемента `K_Г[a, b]`.
"""
function K_Г_ab(a::Int, b::Int, c::Counts, b_body::Float64, boundary::Boundary)
    res = 0.0

    if a == b
        if a ≤ c.Ns
            (i, k) = inverse_body_index(a, c)
            k == 1 || return 0.0
            for j in 1:c.Nc, m in 1:boundary.M
                res += g_ijm(i, j, m, b_body, boundary)
            end
        else
            j = inverse_channel_index(a, c)
            for i in 1:c.Nx, m in 1:boundary.M
                res += g_ijm(i, j, m, b_body, boundary)
            end
        end
    elseif (a ≤ c.Ns) ⊻ (b ≤ c.Ns)
        ((i, k), j) = a ≤ c.Ns ? (inverse_body_index(a, c), inverse_channel_index(b, c)) : 
                                 (inverse_body_index(b, c), inverse_channel_index(a, c))
        k == 1 || return 0.
        for m in 1:boundary.M
            res -= g_ijm(i, j, m, b_body, boundary)
        end
    end

    return res
end

"""
    K_c_ab(a, b, c, β) -> Float64

Элемент `(a, b)` матрицы `K_c` — адвективного переноса в канале
(противопоточная схема при `β > 0`).

Для ячеек канала (`a, b > Ns`):

- `a = b`     : `K_c_ab = +β`, диагональный вклад ячейки `j = a - Ns`;
- `a - b = 1` : `K_c_ab = -β`, связь ячейки `j = a - Ns` с входом
  предыдущей ячейки `j - 1` (только при `j > 1`; для `j = 1` вход
  идёт через правую часть `f`, а не через матрицу);
- иначе       : `K_c_ab = 0`.

Если хотя бы один из индексов относится к детали (`a ≤ Ns` или `b ≤ Ns`),
элемент равен нулю.

# Аргументы
- `a`, `b` : сквозные индексы, `1..Nz`.
- `c`      : `Counts` с полем `Ns`.
- `β`      : тепловой эквивалент расхода, Вт/К.

# Возвращает
`Float64` — значение элемента `K_c[a, b]`.
"""
function K_c_ab(a::Int, b::Int, c::Counts, β::Float64)
    res = 0.0

    if (a > c.Ns) && (b > c.Ns)
        if a == b
            res += β
        elseif a - b == 1
            res -= β
        end
    end

    return res
end

"""
    K_ab(a, b, c, boundary, params, conductivity) -> Float64

Элемент `(a, b)` полной матрицы `K = K_s + K_a + K_Г + K_c`.

    K_ab = K_s_ab(a, b, gx, gy, c)
         + K_a_ab(a, b, gxa, gya, c)
         + K_Г_ab(a, b, c, b_body, boundary)
         + K_c_ab(a, b, c, β)

Проводимости `gx, gxa, gy, gya` берутся из `conductivity` — они не
зависят от `a, b` и считаются один раз.

# Аргументы
- `a`, `b`        : сквозные индексы, `1..Nz`.
- `c`             : `Counts` с полями `Nx`, `Ny`, `Ns`, `Nc`.
- `boundary`      : `Boundary` с полями `M`, `h`, `ell`.
- `params`        : `PhysicsParam`; используются `b`, `β`.
- `conductivity`  : `Conductivity` с полями `gx`, `gxa`, `gy`, `gya`.

# Возвращает
`Float64` — значение элемента `K[a, b]`.
"""
function K_ab(a::Int, b::Int, c::Counts, boundary::Boundary, params::PhysicsParam, conductivity::Conductivity)
    return K_s_ab(a, b, conductivity.gx, conductivity.gy, c) +
        K_a_ab(a, b, conductivity.gxa, conductivity.gya, c) +
        K_Г_ab(a, b, c, params.b, boundary) + 
        K_c_ab(a, b, c, params.β)

end

"""
    K_matvec(vec, c, boundary, params, body_grid) -> Vector{Float64}

Умножает вектор температур `vec` длины `Nz = Ns + Nc` на матрицу `K`
полудискретной системы `M * dx/dt + K * x = f(t)`.

`K = K_s + K_a + K_Г + K_c` вычисляется как четыре независимых прохода

# Аргументы
- `vec`        : вектор длины `Nz = Ns + Nc`.
- `c`          : `Counts` с полями `Nx`, `Ny`, `Ns`, `Nc`, `Nz`.
- `boundary`   : `Boundary` с полями `M`, `h`, `ell`.
- `params`     : `PhysicsParam`; используются `k_s`, `h_a`, `b`, `β`.
- `body_grid`  : `GridBody` с полями `dx`, `dy`, `Vs`; используется
  в `Conductivity` для проводимостей.

# Возвращает
Новый вектор `K * vec` длины `Nz`.
"""
function K_matvec(vec::Vector{Float64}, c::Counts, boundary::Boundary,
                  params::PhysicsParam, body_grid::GridBody)
    Nz = length(vec)
    res = zeros(Float64, Nz)
    cond = Conductivity(params, body_grid)

    # K_s: диффузия детали
    for k in 1:c.Ny, i in 1:c.Nx
        a = body_index(i, k, c)
        if i < c.Nx
            b = body_index(i + 1, k, c)
            d = cond.gx * (vec[a] - vec[b])
            res[a] += d
            res[b] -= d
        end
        if k < c.Ny
            b = body_index(i, k + 1, c)
            d = cond.gy * (vec[a] - vec[b])
            res[a] += d
            res[b] -= d
        end
    end

    # K_a: внешняя теплоотдача на трёх сторонах
    for k in 1:c.Ny, i in 1:c.Nx
        a = body_index(i, k, c)
        ga = (i == 1 || i == c.Nx) * cond.gxa + (k == c.Ny) * cond.gya
        res[a] += ga * vec[a]
    end

    # K_Г: контакт. Один проход по boundary.ell
    for ((i, j, m), ℓ) in boundary.ell
        g = params.b * boundary.h[m] * ℓ
        a = body_index(i, 1, c)
        b = channel_index(j, c)
        d = g * (vec[a] - vec[b])
        res[a] += d
        res[b] -= d
    end

    # K_c: перенос в канале
    for j in 1:c.Nc
        b = channel_index(j, c)
        res[b] += params.β * vec[b]
        if j > 1
            res[b] -= params.β * vec[channel_index(j - 1, c)]
        end
    end

    return res
end

"""
    K_matvec_par(vec, c, boundary, params, body_grid) -> Vector{Float64}

Параллельный вариант `K_matvec`: умножение вектора температур на матрицу `K`.

# Аргументы
- `vec`       : вектор длины `Nz = Ns + Nc`.
- `c`         : `Counts`.
- `boundary`  : `Boundary`.
- `params`    : `PhysicsParam`.
- `body_grid` : `GridBody`.

# Возвращает
Новый вектор `K * vec` длины `Nz`.
"""
function K_matvec_par(vec::Vector{Float64}, c::Counts, boundary::Boundary,
                      params::PhysicsParam, body_grid::GridBody)
    res = zeros(Float64, c.Nz)
    cond = Conductivity(params, body_grid)
    Ns = c.Ns

    # K_s + K_a: один параллельный цикл по ячейкам детали
    Threads.@threads for a in 1:Ns
        (i, k) = inverse_body_index(a, c)

        s = 0.0
        if i < c.Nx
            s += cond.gx * (vec[a] - vec[a + 1])
        end
        if i > 1
            s += cond.gx * (vec[a] - vec[a - 1])
        end
        if k < c.Ny
            s += cond.gy * (vec[a] - vec[a + c.Nx])
        end
        if k > 1
            s += cond.gy * (vec[a] - vec[a - c.Nx])
        end

        ga = (i == 1 || i == c.Nx) * cond.gxa + (k == c.Ny) * cond.gya
        s += ga * vec[a]

        res[a] = s
    end

    # K_Г
    # for ((i, j, m), ℓ) in boundary.ell
    #     g = params.b * boundary.h[m] * ℓ
    #     a = body_index(i, 1, c)
    #     b = channel_index(j, c)
    #     d = g * (vec[a] - vec[b])
    #     res[a] += d
    #     res[b] -= d
    # end
    triples = collect(boundary.ell)
    N = length(triples)
    nt = min(Threads.nthreads(), N)
    bufs = [zeros(Float64, c.Nx + c.Nc) for _ in 1:nt]
    chunk_size = cld(N, nt)

    Threads.@threads for chunk in 1:nt
        buf = bufs[chunk]
        lo = (chunk - 1) * chunk_size + 1
        hi = min(chunk * chunk_size, N)
        for idx in lo:hi
            ((i, j, m), ℓ) = triples[idx]
            g = params.b * boundary.h[m] * ℓ
            d = g * (vec[i] - vec[c.Ns + j])
            buf[i] += d
            buf[c.Nx + j] -= d
        end
    end

    for chunk in 1:nt
        buf = bufs[chunk]
        for i in 1:c.Nx
            res[i] += buf[i]
        end
        for j in 1:c.Nc
            res[c.Ns + j] += buf[c.Nx + j]
        end
    end

    # K_c
    Threads.@threads for j in 1:c.Nc
        b = channel_index(j, c)
        res[b] += params.β * vec[b]
        if j > 1
            res[b] -= params.β * vec[channel_index(j - 1, c)]
        end
    end

    return res
end

"""
    compute_right_side_function(c, body_grid, params, channel, src, conductivity)
        -> Vector{Function}

Строит правую часть `f(t)` полудискретной системы `M * dx/dt + K * x = f(t)`.

Возвращает вектор длины `Nz`, каждый элемент — функция времени `t`.

Компоненты:

- ячейка детали (`a ≤ Ns`, ячейка `(i, k)` в точке `(xc[i], yc[k])`):
    `f_a(t) = Vs * q(x, y, t) + (ga_x + ga_y) * Ta(t)`
  где
    `Vs = b * dx * dy`,
    `ga_x = gxa * ([i = 1] + [i = Nx])`,
    `ga_y = gya * [k = Ny]`,
    `gxa = h_a * dy * b`, `gya = h_a * dx * b` (из `Conductivity`).
  Вклад `Ta(t)` присутствует только для ячеек на трёх наружных сторонах
  детали (левая, правая, верхняя). В углу складываются обе стороны.

- первая ячейка канала (`a = Ns + 1`):
    `f_a(t) = β * Tin(t)`.

- остальные ячейки канала (`a > Ns + 1`):
    `f_a(t) = 0`.

# Аргументы
- `c`             : `Counts` с полями `Ns`, `Nz`, `Nx`, `Ny`.
- `body_grid`     : `GridBody` с полями `Vs`, `xc`, `yc`.
- `params`        : `PhysicsParam`; используется `β`.
- `channel`       : `FlowChannel` с полем `Tin`.
- `src`           : `Source` с полями `q` и `Ta`.
- `conductivity`  : `Conductivity` с полями `gxa`, `gya`.

# Возвращает
`Vector{Function}` длины `Nz`. Каждый элемент — функция `t -> Float64`.
"""
function compute_right_side_function(c::Counts, body_grid::GridBody,
                                     params::PhysicsParam, channel::FlowChannel,
                                     src::Source, conductivity::Conductivity)
    Vs = body_grid.Vs
    gxa = conductivity.gxa
    gya = conductivity.gya

    res = Vector{Function}(undef, c.Nz)
    fill!(res, t -> 0.0)    

    for a in 1:c.Nz
        if a ≤ c.Ns
            (i, k) = inverse_body_index(a, c)
            x = body_grid.xc[i]
            y = body_grid.yc[k]
            ga_x = gxa * ((i == 1) + (i == c.Nx))
            ga_y = gya * (k == c.Ny)
            res[a] = t -> Vs * src.q(x, y, t) + (ga_x + ga_y) * src.Ta(t)
        elseif a == c.Ns + 1
            res[a] = t -> params.β * channel.Tin(t)
        end
    end

    return res
end

"""
    compute_right_side_function_par(c, body_grid, params, channel, src, conductivity)
        -> Vector{Function}

Параллельный вариант `compute_right_side_function`: вектор правой части `f(t)`.

# Аргументы
- `c`            : `Counts`.
- `body_grid`    : `GridBody`.
- `params`       : `PhysicsParam`.
- `channel`      : `FlowChannel`.
- `src`          : `Source`.
- `conductivity` : `Conductivity`.

# Возвращает
`Vector{Function}` длины `Nz`.
"""
function compute_right_side_function_par(c::Counts, body_grid::GridBody,
                                     params::PhysicsParam, channel::FlowChannel,
                                     src::Source, conductivity::Conductivity)
    Vs = body_grid.Vs
    gxa = conductivity.gxa
    gya = conductivity.gya

    res = Vector{Function}(undef, c.Nz)
    fill!(res, t -> 0.0)    

    Threads.@threads for a in 1:c.Ns
        (i, k) = inverse_body_index(a, c)
        x = body_grid.xc[i]
        y = body_grid.yc[k]
        ga_x = gxa * ((i == 1) + (i == c.Nx))
        ga_y = gya * (k == c.Ny)
        res[a] = t -> Vs * src.q(x, y, t) + (ga_x + ga_y) * src.Ta(t)
    end
    res[c.Ns + 1] = t -> params.β * channel.Tin(t)

    return res
end

end