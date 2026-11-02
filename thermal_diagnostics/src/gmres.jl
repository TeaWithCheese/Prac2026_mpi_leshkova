using OhMyThreads
import LinearAlgebra: UpperTriangular, ldiv!

export gmres

"""
    gmres(operator, rhs, z)

Решение системы `A x = b` обобщённым методом минимальных невязок

# Аргументы
- `operator` — функция умножения матрицы `A` на вектор
- `rhs` — правая часть `b`
- `z` — начальное приближение
# Параметры
- `atol` — абсолютная погрешность (εₐ = 10⁻¹²)
- `rtol` — относительная погрешность (εᵣ = 10⁻¹⁰)
- `maxiter` — ограничение числа итераций (3000)
- `restart` — период рестартов (m = 30)
- `htol` — допуск вырождения крыловского пространства (10⁻¹²)

# Исключения
- `LinearAlgebra.SingularException` — вырожденная треугольная система
- `ConvergenceException(x)` — предел числа итераций достигнут на приближении x
"""
function gmres(
    operator,
    rhs::AbstractVector,
    z::AbstractVector;
    atol=1e-12,
    rtol=1e-10,
    maxiter=3000,
    restart=30,
    htol=1e-12,
)
    T = eltype(z)
    n = length(z)
    iter = 0
    restart = min(restart, n, maxiter)
    rhsnorm = norm(rhs)
    tol = atol + rtol * rhsnorm

    r = similar(rhs)
    V = Matrix{T}(undef, n, restart)
    R = Matrix{T}(undef, restart, restart)
    Q = Vector{GivensRotation{T}}()
    y = Vector{T}(undef, restart + 1)
    z = copy(z)

    while iter < maxiter
        r .= rhs .- operator(z)
        ρ = norm(r)
        if ρ <= tol
            return z
        end
        @. V[:, 1] = r / ρ
        resize!(Q, 0)
        jlast = restart
        for j = 1:restart
            iter += 1
            w = operator(@view V[:, j])
            for i = 1:j
                vi = @view V[:, i]
                hij = R[i, j] = dot(vi, w)
                muladd!(w, vi, -hij)
            end
            hlast = norm(w)
            hbreak = abs(hlast) <= htol
            if !hbreak && j < restart
                invhlast = inv(hlast)
                @views mul!(V[:, j + 1], w, invhlast)
            end

            # обновление Q, R
            @views apply!(R[:, j], Q)
            G, rjj = givens_elimination(R[j, j], hlast)
            R[j, j] = rjj
            push!(Q, G)

            y[1] = ρ
            y[2:j + 1] .= zero(T)
            apply!(y, Q) # ȳ ← Qρe₁
            @views ldiv!(UpperTriangular(R[begin:j, begin:j]), y[begin:j])
            if abs(y[j + 1]) <= tol # |ȳⱼ₊₁| — теоретическая невязка
                x = copy(z)
                @views muladd!(x, V[:, begin:jlast], y[begin:jlast])
                r .= rhs .- operator(x)
                if norm(r) <= tol
                    return x
                end
                # продолжить без рестарта
            end
            if hbreak
                jlast = j
                break
            end
        end
        @views muladd!(z, V[:, begin:jlast], y[begin:jlast])
    end
    throw(ConvergenceException(z))
end

"""
    dot(x, y)

Параллельно вычисляемое скалярное произведение
"""
dot(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number} =
    tmapreduce(
        (x, y) -> x * y, +, x, y;
        init=zero(T), scheduler=:static,
    )

"""
    norm(v)

Параллельно вычисляемая евклидова норма вектора
"""
norm(v::AbstractVector{T}) where {T <: Number} =
    sqrt(tmapreduce(
        x -> x^2, +, v;
        init=zero(T), scheduler=:static,
    ))

"""
    mul!(y, x, α)

Параллельное присваивание y=αx
"""
function mul!(
    y::AbstractVector{T}, x::AbstractVector{T}, α::T,
) where {T <: Number}
    Threads.@threads for i in eachindex(x)
        y[i] = x[i] * α
    end
    y
end

"""
    muladd!(C, A, B)

Параллельное прибавление произведения C+=AB
"""
function muladd!(
    y::AbstractVector{T}, x::AbstractVector{T}, α::T=one(T),
) where {T <: Number}
    Threads.@threads for i in eachindex(x)
        y[i] += x[i] * α
    end
    y
end
function muladd!(
    y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, 
) where {T}
    for (j, (v, α)) in enumerate(zip(eachcol(A), x))
        muladd!(y, v, α)
    end
    y
end

"""
    GivensRotation{T}(cos, sin)

Оператор вращения Гивенса для двух строк,
соответствует матрице `[cos sin; -sin cos]`.
"""
struct GivensRotation{T <: Number}
    cos::T
    sin::T
end

"""
    givens_elimination(a, b)

Построение вращения Гивенса для обнуления элемента `b` при помощи элемента `a`

# Возвращаемое значение
Пара `(G, r)`: G — искомый оператор, `G [a, b] = [r, 0]`
"""
function givens_elimination(a::T, b::T) where {T <: Number}
    r = hypot(a, b)
    if iszero(r)
        return GivensRotation{T}(one(T), zero(T)), r
    end
    GivensRotation{T}(a / r, b / r), r
end

"""
    apply!(v, G)

Применение вращения Гивенса к двухэлементному вектору `v`
"""
function apply!(v::AbstractVector{T}, G::GivensRotation{T}) where {T <: Number}
    x = v[1]
    y = v[2]
    v[1] = G.cos * x + G.sin * y
    v[2] = G.cos * y - G.sin * x
    v
end

"""
    apply!(v, Q)

Применение композиции вращений Гивенса `Q` к вектору `v`.
Операторы Q₁, …, Qₙ применяются последовательно к парам (v₁, v₂), …, (vₙ₋₁, vₙ).
"""
function apply!(v::AbstractVector{T}, Q::AbstractVector{GivensRotation{T}}) where {T <: Number}
    for (i, G) in enumerate(Q)
        @views apply!(v[i:i + 1], G)
    end
    v
end

"""
    ConvergenceException(lastapprox)

Неудача итерационного метода.
За выделенное число шагов метод не достиг заданной точности.
`lastapprox` — последнее полученное приближение к решению.
"""
struct ConvergenceException{T} <: Exception
    lastapprox::T
end

Base.showerror(io::IO, e::ConvergenceException) =
    print(io, "Не удалось достичь заданной точности")
