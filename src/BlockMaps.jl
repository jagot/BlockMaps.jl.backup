module BlockMaps
using LinearMaps
using RecipesBase

struct Block{T}
    a::AbstractMatrix{T}
    i::Integer
    j::Integer
end

Base.eltype(b::Block{T}) where T = T
Base.size(b::Block) = size(b.a)

Base.getindex(b::Block, I::Union{Integer,UnitRange}, J::Union{Integer,UnitRange}) =
    b.a[I-b.i+1,J-b.j+1]
Base.setindex!(b::Block{T}, v::T, I::Union{Integer,UnitRange}, J::Union{Integer,UnitRange}) where T =
    b.a[I-b.i+1,J-b.j+1] = v


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
    clear_overlaps::Bool
    overlap_tol::Float64
end

# properties
Base.size(A::BlockMap) = (A.m,A.n)
Base.issymmetric(A::BlockMap) = A._issymmetric
Base.ishermitian(A::BlockMap) = A._ishermitian
Base.isposdef(A::BlockMap) = A._isposdef

function (::Type{BlockMap})(::Type{T}, m::Integer, n::Integer;
                            issymmetric::Bool = false,
                            ishermitian::Bool = false,
                            isposdef::Bool = false,
                            clear_overlaps::Bool = false,
                            overlap_tol::Float64 = eps(Float64)) where {T}
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
             issymmetric, ishermitian, isposdef,
             clear_overlaps, overlap_tol)
end

function (::Type{BlockMap})(m::Integer, n::Integer,
                            indices::Vector{Tuple{Integer,Integer}},
                            blocks::Vector{M};
                            kwargs...) where M<:AbstractMatrix{T} where T
    B = BlockMap(eltype(first(blocks)), m, n; kwargs...)
    for i in eachindex(indices)
        B[indices[i]...] = blocks[i]
    end
    B
end

function (::Type{BlockMap})(indices::Vector{Tuple{Integer,Integer}},
                            blocks::Vector{M};
                            kwargs...) where M<:AbstractMatrix{T} where T
    m,n = 0,0
    for (i,b) in enumerate(blocks)
        bm,bn = size(b)
        m = max(m, indices[i][1]+bm-1)
        n = max(n, indices[i][2]+bn-1)
    end
    BlockMap(m,n, indices, blocks; kwargs...)
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
            overlap = t(nb, b)
            if !any(isempty.(overlap))
                if !A.clear_overlaps
                    error("Cannot insert new $(nb) overlapping with old block at $(b)")
                else
                    d = vecnorm(nb[overlap...]-b[overlap...])
                    if d > A.overlap_tol
                        error("Overlapping regions of $(nb) and $(b) differ by $(d) > $(A.overlap_tol)")
                    end
                    b[overlap...] = zero(T)
                end
            end
        end
    end

    push!(A.blocks, nb)
    nb
end

# multiplication with vector
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

@recipe function plot(A::BlockMap)
    legend --> false
    yflip --> true
    seriestype := :heatmap
    aspect_ratio := 1
    full(A)
end

export BlockMap

end # module
