# Tensars.jl

This package exports a `Tensar` type, which can be identified with
tensors in the mathematical sense, but represents them in a different
way than mathematicians customarily define them.

A `tensar` is a linear mapping from m-dimensional arrays to n-dimensional
ones.  For example, let `D2` be an `5×5` matrix that discretises
the second derivative operator, and `u` be a 5×5×5 matrix that
samples a field on a 3-dimensional grid.  Then the Laplacian of the
field can be computed as follows.

    using Tensars, LinearAlgebra
    
    julia> D2 = Tensar(D2)
    5-vector → 5-vector Tensar{Float64}
    
    julia> E = Tensar(Float64.(Matrix(I,5,5)))
    5-vector → 5-vector Tensar{Float64}
    
    julia> L = D2⊗E⊗E + E⊗D2⊗E + E⊗E⊗D2
    5×5×5 → 5×5×5 Tensar{Float64}
    
    julia> Lu = L*u;
    
    julia> typeof(Lu)
    Array{Float64,3}

The way to identify a `Tensar` with a mathematical tensor is specified
below, in the section *Mathematical tensors and tensor products*.
The motivation for defining it as a mapping of arrays is that those
mappings are closed under composition.

This is research software, and the hypothesis is that tensars will
be widely useful generalisation of matrices.  Here is another
example.

    julia> using ForwardDiff
    
    julia> v = randn(2,3,4);
    
    julia> f(u) = [sum(u)]
    
    julia> J_matrix = ForwardDiff.jacobian(f, v)
    
    julia> J = reshape(J_matrix, size(f(v)), size(v))
    2×3×4 → 1-vector Tensar{Float64}
    
    julia> dv = ones(2,3,4);
    
    julia> J*dv
    1-element Array{Float64,1}:
     24.0

This package could be regarded as a port of Sussman and Wisdom's up and
down tuples, replacing the parentheses of Scheme with the brackets
of Julia.  No doubt I have botched it, in which case I apologise
for messing up their design.
  
The current implementation is a proof of concept.  A production
version would implement eltype promotion, and treat `I` as a
broadcastable identity operator, avoiding the need to specify `E`:

    julia> L = D2⊗I⊗I + I⊗D2⊗I + I⊗I⊗D2

The storage required for a dense n-dimensional `Tensar` increases
geometrically with n, so these will require special types for
structured forms.  One use might be to accumulate one-dimensional
convolutions into a multidimensional operator, in order to evaluate
them a cache-efficient way with a machine learning library.  Another
way to regard this package is as a refactoring of `DiffEqOperators`
that went completely overboard.

## The Julia tensor ecosystem

There are several other packages dealing with tensors.  Eric Forgy has [reviewed](https://ericforgy.github.io/TensorAlgebra.jl/dev/) them in detail, so here we'll focus on how the relate to Tensars.

There are two major ways that Tensars is different:

1. The usual approach is to take the textbook mathematical notation
and semantics for tensors, and try to import as much of them into
Julia as possible.  Tensars goes the other way.  The goal is to
extend Julia's arrays and unilinear algebra in a consistent fashion,
and make it convenient for programs to do multilinear algebra.  The
result can be identified with the mathematical treatment of tensors,
but it looks quite different.

2. Most tensor libraries are written with differential geometry,
curvature and general relativity in mind, but this one was written
with quantum mechanics in mind.  This has some pros and cons.  The
benefit is that complex numbers and Hilbert spaces are more than
an afterthought.  They have just as natural an interface as do real
numbers and Euclid or Lorentz spaces, and they're just as thoroughly
tested.  (Does anyone work with in complex spaces where vectors can
have negative norms?)

The drawback is that index raising and lowering is less convenient.
These operations are linear in the real case, but not in the complex
case, and Tensars doesn't provide a convenient way to exploit that
linearity.  If you're au fait with relativistic index bashing, and
you see an obvious way to incorporate it in the tensar formalism,
please file an issue.

## Construction and linear algebra

TODO `Tensar(::scalar)` will construct a UniformTensar.  `Tensar(::scalar, cs, rs)` has a shape.

TODO What about `5×6 → 2×0×4 Tensar`?

The simplest way to construct a `Tensar` is to reshape an array
with a column size and a row size.  Following the convention for
vectors and matrices, `x` and `y` are arrays, while `A` and `B` are
tensars.

    julia> A_matrix = rand(2*3*4, 5*6);
    
    julia> A = reshape(A_matrix, (2,3,4), (5,6))
    5×6 → 2×3×4 Tensar{Float64}
    
    julia> typeof(A)
    Tensar{Float64,3,2}
    
    julia> Array(A)
    2×3×4×5×6 Array{Float64,5}:
    ...

The linear mapping `A*x` is what anyone who works with 3D fields
will expect it to be.

    julia> x = rand(5, 6);
    
    julia> A*x ≈ reshape(A_matrix*x[:], (2,3,4))
    true

By convention, when a function that returns some property of an
`Array` is also defined for for `Tensar`, it returns the same
property of the arrays that form the column and row spaces of
the `Tensar` as a 2-tuple.

    julia> colsize(A)
    (2, 3, 4)
    
    julia> colsize(A,2)
    3
    
    julia> rowsize(A)
    (5, 6)
    
    julia> size(A)
    ((2, 3, 4), (5, 6))
    
    julia> ncols(A)
    3
    
    julia> nrows(A)
    2
    
    julia> ndims(A)
    (3, 2)

Following this convention, the `length` of a `Tensar` is the size
of the matrix for the linear transformation that it represents.

    julia> length(A)
    (24, 30)
    
    julia> ans == size(A_matrix)
    true

The constructor `Tensar(::AbstractArray)` usually constructs a
column tensar, that maps a scalar to a scalar multiple of the array.

    julia> ket = Tensar(rand(2,3,4))
    scalar → 2×3×4 Tensar{Float64}

The exception is that `Matrix` is treated as a linear transformation.

    julia> M = rand(3,4);
    
    julia> Tensar(M)
    4-vector → 3-vector Tensar{Float64}

So the general way to construct a column tensar is:

    julia> Tensar(M, ndims(M), 0)
    scalar → 3×4 Tensar{Float64}

The constructor also accepts `nrows` and `ncols`.

    julia> Tensar(rand(4,5,6), 1, 2)
    5×6 → 4-vector Tensar{Float64}

The adjoint of a column tensar is a row tensar, which maps arrays
to scalars according to `LinearAlgebra.dot`.

    julia> bra = ket'
    2×3×4 → scalar Tensar{Float64}
    
    julia> x = rand(2,3,4);
    
    julia> bra*x ≈ Array(ket)⋅x
    true
    
    julia> Tensar(rand(5)')
    5-vector → scalar Tensar{Float64}

Just as Julia supports 0-dimensional arrays with a single element,
tensars can have shape 0,0.  This requires some choices about which
products return scalars, arrays and tensars.  The rules are stated
below.

For now, indexing tensars simply indexes into the array of their
elements.  This is the simplest way to do it, but I'm not convinced
it is the right way, and it might change.

It is common to think of a matrix shape as “rows × columns”, but
this becomes confusing when generalised to tensars.  A matrix maps
its row space to its column space, and `Tensar{T,2,3}` maps
3-dimensional arrays to 2-dimensional ones, so it has 3 row dimensions
and 2 column ones.  It can be particularly confusing that `size(A)
== ((2, 3, 4), (5, 6))`, while `A` displays as a `5×6 → 2×3×4` mapping,
but the alternatives all seem worse.  Users are advised to toss
“number of rows × number of columns” down the nearest memory hole,
and start thinking “column length × row length”.

Although the elements of a `Tensar` form an array, and `Array(Tensar(x))
== x` identically, `Tensar` is not a subtype of `AbstractArray`.
Every matrix can be identified with a 1,1-dimensional tensar, but
there are 2,0- and 0,2-dimensional tensars that have the same
elements but represent different linear transformations.  Similarly,
every vector can be identified with a column tensar, but there is
a distinct row tensar with the same elements.

Tensars form a linear algebra over their element type in exactly
the same way that matrices do.  They can be added just like matrices,
and both `*` and `⊗` reduce to the scalar product when one operand is a
scalar.

## The rules on array and tesnar shapes

When an array is reshaped to a tensar, there are two rules on the
array and tesnar shapes.  Suppose the tensar shape is `a×b×c←d×(e*f)×g`.

1. The tensor sizes divide the array sizes.  This tensar could be
formed from an `(a,b,c,d,e*f,g)` array, an `(a*b*c, d*e*f*g)` array,
or any intermediate choice of commas and multiplications.  But not
from an `(a,b,c,d,e,f,g)` array, or a `(1,a*b*c,d*e*f*g)` array.

2. The array size can be split into a prefix that divides the column
length of the tensar, and a suffix that divides its row length.
This excludes an `(a*b,c*d,e*f*g)` array, for example.

When the array is 0-dimensional, or the tensar is a row, column or
scalar, only `()` and `scalar` are deemed to divide `()` and `scalar`.
A 0-dimensional array has a single element, but it can only form a
`scalar ← scalar` tensar, not `scalar ← 1-vector` or `1×1 ← scalar`
or whatever.

These rules are relaxed slightly for constructing `TensArray`.
Scalar row or column sizes can correspond to a single array dimension,
in order that every shape of `Tensar` can be a `TensArray` matrix.

## Generalising the matrix product

There are two product operators that act on `Tensar`.  The tensor
product `⊗` has its usual mathematical meaning, which will be
discussed below.  Note that neither `*` nor `⊗` is not commutative.

The operator product `*` is identical to the matrix product, when
matrices and tensars are both identified with linear mappings.  The
product `A*x` is the image of the array `x` under the linear mapping
`A`.  The product `A*B` is the composition of the linear mappings
`A` and `B`.  If `x` is a vector, then `x'*A` is the same as
`x'*Array(A)`.

Here are the full rules on which products return scalars, arrays
and tensars.  The tensar library treats anything other than an
`AbstractArray` or `AbstractTensar` as a scalar.

1. The product of two tensars is always a tensar.  In particular,
this returns a `scalar ← scalar Tensar` instead of a scalar.

2. The product of a scalar and a tensar is a scalar product, and
returns a tensar.

3. The product of a tensar and an array follows the Julia rules for
matrix multiplication.   The result is usually an array, except
that a scalar is returned instead of a 0-dimensional array.

## Adjoints

Tensars have adjoints, like any linear mappings.  Row and column
`Tensars` have the same adjoint relationship as row and column
vectors, or bras and kets in Dirac notation.

    Tensar(A, n, 0)' == Tensar(conj.(A), 0, n)

In general, `A'` is the unique tensar that satisfies the identity

    V'*A'*U' = conj(U*A*V)

where `U` is any appropiately shaped row tensar, and `V` any
appropiately shaped column tensar.

## Tensors and tensor products

It was mentioned at the beginning that tensars can be identified
with tensors.  The time has come to describe this correspondence
and the tensor product `⊗`, which acts on tensars the same way it acts
on the corresponding tensors.

Mathematicians (well, OK, physicists) conventionally define an
m,n-tensor as a multilinear mapping from `m` dual vectors and `n`
vectors to a scalar.  This will be identified with a certain
m,n-dimensional tensar.  A symbol such as `A` could denote either
of these, but the tensor mapping will be written as a function
application `A(u1, ..., um; v1, ..., vn)` and the tensar mapping as `A*x`.

The tensor product has a very simple definition in terms of tensors.
Tensors are scalar valued, and `⊗` just multiplies those scalars.

    (A⊗B)(u1, ..., um; v1, ..., vn) =
        A(u1, ..., uj; v1, ..., vk) × B(u[j+1], ..., um; v[k+1], ..., vn)

The correspondence between tensors and tensars can be built up
using the tensor product, starting from basis vectors, then progressing
to row and column tensars, and finally to general tensars.

A dual vector `v'` can be identified with a 0,1-tensor in an obvious
way, as the mapping `u -> v'⋅u`.  Similarly, a vector `u` can be
identified with the 1,0-tensor `v -> v⋅u`.  These tensors are
identified with `Tensar(u)` and `Tensar(v')`.

A vector is identified with a tensor that takes a dual vector
argument, so a tensor product of vectors is a tensor with only dual
vector arguments.  If `e[j]` is the usual basis vector, an array
can be formed by tabulating

    A[j, k, ...] = (e[j]⋅v1)×(e[k]⋅v2)×... = (e[j]⊗e[k]⊗...)(v1, v1, ...)

The tensor `e[j]⊗e[k]⊗...` corresponds to `Tensar(A)`.

The product of 0,1-dimensional tensars can be written in terms of their
1,0-dimensional adjoints:

   (v1'⊗v2'⊗...) = (v1⊗v2⊗...)'

This is consistent with the definition of `⊗` in terms of tensors.
We now have the machinery to identify a general `Tensar` with a
tensor:

    A(u₁, ..., u_m, v₁, ..., v_n) = (u₁⊗...⊗u_m)*A*(v₁⊗...⊗v_n)

The action of `⊗` on general tensars is now determined by identifying
them with tensors.

The trace of a `Tensar` is a tensor contraction.  It takes two
arguments, the column and row dimensions to contract over.  These
must have equal lengths.  Index raising works as follows.  When
`permutedims` is implemented, it will be possible to shuffle the
raised index into an appropriate place.

    julia> A = Tensar(rand(2,3,4,5,6,7), 3, 3)
    5×6×7 → 2×3×4 Tensar{Float64}
    
    julia> g = Tensar(rand(6,6),2,0)
    scalar → 6×6 Tensar{Float64}
    
    julia> g⊗A
    5×6×7 → 6×6×2×3×4 Tensar{Float64}
    
    julia> tr(g⊗A,1,2)
    5×7 → 6×2×3×4 Tensar{Float64}

Contraction has always looked like a matrix product.  The `Tensar`
formalism makes apparent how this corresponds to a composition of
linear mappings (except that the implementation currently has a
bug, so these don't come out equal):

    julia> M = Tensar(rand(4,5))
    5-vector → 4-vector Tensar{Float64}

    julia> N = Tensar(rand(5,6))
    6-vector → 5-vector Tensar{Float64}
    
    julia> M⊗N
    5×6 → 4×5 Tensar{Float64}
    
    julia> tr(M⊗N, 2, 1)
    6-vector → 4-vector Tensar{Float64}
    
    julia> M*N
    6-vector → 4-vector Tensar{Float64}

I'm not quite satisfied with this.  I think that, lurking somewhere on
the edge of it, there are some interesting ideas about inner products
of arrays and `g^{ij}` being the inverse of `g_{ij}`.  If you know
enough multilinear algebra to see that clearly, please explain
it to me.

## TensArrays

Arrays are the currency of Julia.  This means that `ForwardDiff.jacobian`
will continue to return an array for the forseeable future, and the
only way for tensors to gain market share is if they can be bought
and sold for arrays.  Any tensor library has to like that, or lump
it and accept that it tensors will be a walled garden.  (Thanks to
Michael Abbot for pointing that out at an early stage in the
development of Tensars.)

The `TensArrays` package aims to provide tensors unbiquitously,
disguised as arrays.

    julia> J_matrix = ForwardDiff.jacobian(f, v)
    
    julia> using TensArrays
    
julia> J_matrix = TensArray(J_matrix, size(f(v)), size(v))
1×24 TensArray{Float64,2}:
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  …  1.0  1.0  1.0  1.0  1.0  1.0  1.0
     
    julia> J_matrix isa AbstractMatrix
    true
    
    julia> using Tensars
    
    julia> Tensar(J_matrix)
    1-vector ← 2×3×4 Tensar{Float64}

This is a deliberately minimalist library, to reduce the performance
cost and encourage functions like `jacobian` to return a `TensArray`.
As a result, the tensor nature is fragile.  `J_matrix` will revert
to an `Array` if it is broadcast with anything except a scalar or
`TensArray`, or if it is reshaped or broadcast in a way that is
inconsistent with its tensor dimensions.  The intention is that
code that wishes to use the tensor information in a `TensArray`
will convert it to a `Tensar` as soon as possible, or to the author's
favorite way to represent tensors.

    julia> 2J_matrix
    1×24 TensArray{Float64,2}:
     2.0  2.0  2.0  2.0  2.0  2.0  2.0  2.0  …  2.0  2.0  2.0  2.0  2.0  2.0  2.0
     
    julia> ans + rand(size(J_matrix)...)
    1×24 Array{Float64,2}:
     2.634  2.40308  2.28292  2.75842  …  2.1715  2.74704  2.2228  2.66855
     
    julia> reshape(J_matrix, 1, 2, 3, 4)
    TensArray
     
    julia> reshape(J_matrix, 1, 12, 2)
    Array

I'm still thinking about how this should interact with lazy reshaping.
It is unlikely to be implemented reliably until someone thinks
harder about nested array wrappers.

Array dimensions can not be a mixture of row and column
tensor dimensions.

    julia> A_matrix = TensArray(A_matrix, (2,3,4), (5,6));
    
    julia> size(A_matrix)
    (24, 30)
    
    julia> reshape(A_matrix, 6, 4, 30)
    6×4×30 TensArray
    
    julia> reshape(A_matrix, 48, 15)
    48×15 Array

## Future work

Generalise `UniformScaling` to axis expansion.  These are the axes
that the codimensions act along, these are the axes the contradimensions
act along, all the other axes have unspecified length and the
elements are diagonal along them.  When the kernel is a scalar, all
other axes means all axes, and it reduces to uniform scaling.
