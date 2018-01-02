module BlockMaps
using LinearMaps

struct Block{T}
    a::AbstractMatrix{T}
    i::Integer
    j::Integer
end

Base.eltype(b::Block{T}) where T = T
Base.size(b::Block) = size(b.a)

function extents(b::Block{T}) where T
    m,n = size(b)
    b.i+(1:m)-1,b.j+(1:n)-1
end

function Base.show(io::IO, b::Block{T}) where T
    m,n = size(b)
    write(io, "$(m)x$(n) $(T) block at ($(b.i),$(b.j))")
end

function Base.intersect(b1::Block{T},b2::Block{T}) where T
    e1 = extents(b1)
    e2 = extents(b2)
    intersect(e1[1],e2[1]),intersect(e1[2],e2[2])
end

function tintersect(b1::Block{T},b2::Block{T}) where T
    e1 = extents(b1)
    e2 = extents(b2)[[2,1]]
    intersect(e1[1],e2[1]),intersect(e1[2],e2[2])
end

struct BlockMap{T} <: LinearMap{T}
    m::Integer
    n::Integer
    blocks::Vector{Block{T}}
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end

# properties
Base.size(A::BlockMap) = (A.m,A.n)
Base.issymmetric(A::BlockMap) = A._issymmetric
Base.ishermitian(A::BlockMap) = A._ishermitian
Base.isposdef(A::BlockMap) = A._isposdef

# multiplication with vector
function (::Type{BlockMap})(::Type{T}, m::Integer, n::Integer;
                            issymmetric::Bool = false,
                            ishermitian::Bool = false,
                            isposdef::Bool = false) where {T}
    if m != n
        issymmetric && error("Non-square matrices cannot be symmetric")
        ishermitian && error("Non-square matrices cannot be hermitian")
        isposdef && error("Non-square matrices cannot be positive definite")
    else
        if T <: Real
            if issymmetric
                ishermitian = true
            elseif ishermitian
                issymmetric = true
            end
        end
    end
    BlockMap(m, n, Block{T}[],
             issymmetric, ishermitian, isposdef)
end

function Base.setindex!(A::BlockMap{T}, a::AbstractMatrix,
                        i::Integer, j::Integer) where T
    nb = Block(a, i, j)
    m,n = size(A)
    ne = extents(nb)
    if i ∉ 1:m || j ∉ 1:n ||
        ne[1][2] ∉ 1:m || ne[2][2] ∉ 1:n
        error("Trying to assign $(nb), outside of matrix extents $(m)x$(n)")
    end

    if i == j
        !isreal(A) && ishermitian(A) && !ishermitian(a) && error("Diagonal blocks of Hermitian matrices must be Hermitian")
        issymmetric(A) && !issymmetric(a) && error("Diagonal blocks of symmetric matrices must be symmetric")
    elseif ishermitian(A) || issymmetric(A)
        # Non-diagonal block of Hermitian/symmetric matrix may not overlap its transposed block
        !any(isempty.(tintersect(nb, nb))) && error("Cannot insert new $(nb) overlapping its own transpose")
    end

    tests = Function[intersect]
    if issymmetric(A) || ishermitian(A)
        # Test also if block intersects with Hermitian
        # conjugate/transpose of old block
        push!(tests, tintersect)
    end
    for b in A.blocks
        for t in tests
            if !any(isempty.(t(nb, b)))
                error("Cannot insert new $(nb) overlapping with old block at $(b)")
            end
        end
    end

    push!(A.blocks, nb)
    nb
end

function Base.A_mul_B!(y::AbstractVector, A::BlockMap{T}, x::AbstractVector) where T
    y[:] = 0
    for b in A.blocks
        be = extents(b)
        y[be[1]] += b.a*x[be[2]]
        if b.i != b.j && (ishermitian(A) || issymmetric(A))
            f = if ishermitian(A)
                Ac_mul_B
            else
                At_mul_B
            end
            y[be[2]] += f(b.a, view(x, be[1]))
        end
    end
    y
end

export BlockMap

end # module
