module Tensars

export Tensar, nrows, ncols, colsize, rowsize, ⊗, ∗

using TensArrays, LinearAlgebra

"""
    OpenTensar{T}

This is a linear operation along certain dimensions of an array,
that is broadcast along all remaining dimensions.
"""
abstract type OpenTensar{T} end

abstract type AbstractTensar{T,M,N} <: OpenTensar{T} end

"""
    Tensar{T,M,N} <: AbstractTensar{T,M,N}

Linear mapping from `N`- to `M`-dimensional arrays with elements of type `T`.

`Tensar(M::Matrix)` constructs a 1,1-tensor that represents the
same linear mapping as `M` does.

Question: should `Tensar(V::Vector)` treat `V` as a column matrix
and return a 1,1-tensor, or leave it as a vector and return a
1,0-tensor?  The latter is nice for construction with inner products.
It's a pity that `Adjoint` is treated as a rank-2 array; that makes
it a bit messy to allow 0,1-tensors to be constructed from `Tensar(x')`.

Note that the current constructors for `Adjoint` and `Matrix` are
inconsistent with the idea that `Array{T,N}` can be identified with
`Tensar{T,N,0}` for product purposes.
"""
struct Tensar{T,M,N}
    elements::Array{T}

    function Tensar{T,M,N}(A::AbstractArray) where T where M where N
        errmsg =
            if !(M isa Int) || !(N isa Int) || M < 0 || N < 0
                "rowdims and coldims must be non-negative integers"
            elseif ndims(A) != M+N
                "$(ndims(A))D Array can not form $(M)D ← $(N)D Tensar"
            else
                ""
            end

        if !isempty(errmsg)
            throw(DimensionMismatch(errmsg))
        elseif M == N == 0
            A[]
        else
            new(Array(A))
        end
    end
end

# Defined for convenience
RowVector{T} = Union{Adjoint{T, S}, Transpose{T, S}} where 
    S <: AbstractVector{T}
    
Tensar(A::AbstractArray{T}, m::Int, n::Int) where T = Tensar{T, m, n}(A)
Tensar(A, (m, n),) = Tensar(A, m, n)

# Default constructors for scalars, (adjoint) vectors, matrices, and arrays
Tensar(z) = Tensar(z,0,0)
Tensar(z,n::Int,m::Int) = Tensar(reshape([z],()), n, m)
Tensar(x::AbstractVector) = Tensar(x,1,0)
Tensar(x::Adjoint{T, S}) where S <: AbstractVector{T} where T = Tensar(x[:],0,1)
Tensar(M::AbstractMatrix) = Tensar(M,1,1)
Tensar(A::AbstractArray) = Tensar(A,ndims(A),0)

# Array casts
Base.Array(A::Tensar{T,0,1}) where T = transpose(A.elements)
Base.Array(A::Tensar) = A.elements

# Equality
Base.:(==)(A::Tensar, B::Tensar) =
    (size(A) == size(B)) && (A.elements == B.elements)
Base.:≈(A::Tensar, B::Tensar) =
    (size(A) == size(B)) && (A.elements ≈ B.elements)
    
# Eltype

Base.eltype(A::Tensar{T}) where T = T

# Zeros

Base.zero(A::Tensar) = Tensar(zero(A.elements), ndims(A)...)

"""
    zeros(T, colsize, rowsize)

Return a zero Tensar with the size specified by colsize and rowsize.
"""
Base.zeros(T, cs, rs) = reshape(zeros(T, prod(cs)*prod(rs)), cs, rs)
   
# Sizes
ncols(A::Tensar{T,M,N}) where T where N where M = M
nrows(A::Tensar{T,M,N}) where T where N where M = N

"""
    nrows(A::Tensar)
    ncols(A::Tensar)
    ndims(A::Tensar) = (ncols(A), nrows(A))

Return the number of row dimensions of `A`, the number of its column
dimensions, or a tuple of both.
"""
Base.ndims(A::Tensar) = (ncols(A), nrows(A))

"""
    colsize(A::Tensar)
    rowsize(A::Tensar)
    size(A::Tensar) = (colsize(A), rowsize(A))

Return a tuple of two tuples, the first containing the lengths of
the row dimensions of `A`, and the second the lengths of its column
dimensions.
"""
function Base.size(A::Tensar)
    # A.elements because Array(A) might be an adjoint vector with a matrix size
    s = size(A.elements)
    n = ncols(A)
    s[1:n], s[n+1:end]
end

Base.length(A::Tensar) = prod.(size(A))

colsize(A::Tensar) = size(A)[1]
rowsize(A::Tensar) = size(A)[2]
colsize(A::Array) = size(A)
rowsize(A::Array) = ()
colsize(A,j) = colsize(A)[j]
rowsize(A,j) = rowsize(A)[j]

# Display

reprsize(_::Tuple{}) = "scalar"
reprsize(n::Tuple{Int}) = "$(n[1])-vector"
reprsize(ns::Tuple) = join(repr.(ns), "×")
function reprsize(A::Tensar)
    o, i = reprsize.(size(A))
    "$o ← $i"
end

Base.show(_::IO, A::Tensar) = print(repr(A))
Base.repr(A::Tensar{T}) where T = "$(reprsize(A)) Tensar{$(repr(T))}"

# Indexing

function Base.getindex(A::Tensar, ixs...)
    nc, nr = ndims(A)
    elts = A.elements
    mc = ndims(elts[ixs[1:nc]..., fill(:, nr)...]) - nr
    mr = ndims(elts[fill(:, nc)..., ixs[1+nc:end]...]) - nc
    Tensar(elts[ixs...], mc, mr)
end

Base.setindex!(A::Tensar, X::Array, ixs...) =
    setindex!(A.elements, X, ixs...)
    
Base.setindex!(A::Tensar, X::Tensar, ixs...) =
    setindex!(A.elements, X.elements, ixs...)

# Addition

function Base.:+(A::Tensar, B::Tensar)
    if size(A) != size(B)
        asize = reprsize(A)
        bsize = reprsize(B)
        throw(DimensionMismatch("sizes $asize and $bsize do not match" ))
    end
    Tensar(A.elements + B.elements, ndims(A)...)
end

# generalised matrix products
# TODO eltype promotion

Base.reshape(A::Tensar, cs::Tuple, rs::Tuple) = reshape(A.elements, cs, rs)
Base.reshape(A::AbstractArray, cs::Tuple, rs::Tuple) = 
    Tensar(reshape(A, (cs..., rs...)), length(cs), length(rs))

splat(A) = reshape(A.elements, length(A)...)

function Base.:*(A::Tensar, B::Tensar)
    if rowsize(A) != colsize(B)
        c = reprsize(rowsize(A))
        r = reprsize(colsize(B))
        throw(DimensionMismatch("column size $c does not match row size $r" ))
    end
    reshape(splat(A)*splat(B), colsize(A), rowsize(B))
end

array_or_scalar(A::Tensar) = Array(A)
array_or_scalar(A::Array) = A
array_or_scalar(z) = z

Base.:*(A::Tensar, B::Matrix) = A*Tensar(B,2,0) |> array_or_scalar
Base.:*(A::Tensar, B::Array) = A*Tensar(B) |> array_or_scalar
Base.:*(A::Adjoint, B::Tensar) = Tensar(A)*B |> array_or_scalar

Base.:*(x::T, A::Tensar{T}) where T = typeof(A)(x.*A.elements)
Base.:*(A::Tensar{T}, x::T) where T = typeof(A)(A.elements.*x)

# partial contraction

"""
    ∗(A::Tensar, B::Tensar)

This is an extension of the `*` product, broadcast over the excess or
singleton row axes of A and column axes of B.

This is only implemented for x ← (j, ..., k) Tensar ∗ (j, ..., k, ...) Array
"""
function ∗(A::Tensar, x::Array)
    bsize = size(x)[1+nrows(A):end]
    Ax = similar(x, colsize(A)..., prod(bsize))
    y = reshape(x, rowsize(A)..., :)
    for j = 1:size(y, ndims(y))
        Ax[fill(:, ncols(A))..., j] = A*y[fill(:, nrows(A))..., j]
    end
    reshape(Ax, colsize(A)..., bsize...)
end

# outer products

⊗(A::Tensar, B::Tensar) =
    reshape(kron(splat(A), splat(B)), 
        (colsize(A)..., colsize(B)...),
        (rowsize(A)..., rowsize(B)...))

⊗(A::Tensar, z) = A*z
⊗(z, A::Tensar) = z*A
⊗(z, w) = z*w

# Tensar adjoint

LinearAlgebra.adjoint(A::Tensar) =
    reshape(splat(A)', rowsize(A), colsize(A))

"""
    tr(A::Tensar, j, k)
    tr(A::Tensar, (j, k))

Contract the jth column dimension of A with its kth row dimension.
"""
function LinearAlgebra.tr(A::Tensar, j::Int, k::Int)
    ocols = colsize(A)[[i != j for i = 1:ncols(A)]]
    orows = rowsize(A)[[i != k for i = 1:nrows(A)]]
    function slice(i)
        ix = Any[Colon() for _ = 1:sum(ndims(A))]
        ix[j] = ix[ncols(A)+k] = i
        ix
    end
    n = colsize(A,j)
    n == rowsize(A,k) || 
        "Contracted column length $n differs from row length $(rowsize(A,k))" |>
            DimensionMismatch |> throw
    Tensar(sum(A.elements[slice(i)...] for i = 1:n), ncols(A)-1, nrows(A)-1)
end

LinearAlgebra.tr(A::Tensar, jk::Tuple) = tr(A, jk...)

# TensArray casts

function Tensar(x::TensArray)
   xt = reshape(parent(x), x.cs..., x.rs...)
   Tensar(xt, length(x.cs), length(x.rs))
end

TensArrays.TensArray(x::Tensar) =
    TensArray(reshape(Array(x), length(x)), size(x)...)

# include("ExpansionTensars.jl")

end # module
