using Test
using Tensars
using Tensars: RowVector
using TensArrays
using TensArrays: match_shape
using LinearAlgebra

# Random complex arrays and tensars

randc(ns::Dims) = Complex.(randn(ns), randn(ns))/√2
randv(ns::Dims) = Tensar(randc(ns), length(ns), 0)
randt(cs::Dims, rs::Dims) = reshape(randc(cs..., rs...), cs, rs)
randc(ns::Int64...) = randc(ns)
randv(ns::Int64...) = randv(ns)

# General postconditions for successful construction

function explicit_shape(cs, rs, data=randc(cs..., rs...))
    nc = length(cs)
    nr = length(rs)
    b = (data isa AbstractArray) ? data : reshape([data], ())
    
    if match_shape(size(b), cs, rs)
        @test (t = reshape(b,cs,rs)) isa Tensar{eltype(b),nc,nr}
        @test (a = Array(t)) == reshape(b, cs..., rs...)
    else
        @test_throws DimensionMismatch reshape(b,cs,rs)
        return
    end
    
    if ndims(b) == length(cs) + length(rs)
        a = data
    else
        @test_throws DimensionMismatch Tensar(data, nc, nr)
    end
    
    @test t == reshape(b[:], cs, rs)
    if a isa AbstractArray
        @test t == Tensar{eltype(a),nc,nr}(a)
        @test t == Tensar(a,nc,nr)
        @test t == Tensar(a,(nc,nr))
    end
end

implicit_shape(data::TensArray) = nothing

# Adjoint vectors are generally a special case in Julia
function implicit_shape(data::LinearAlgebra.Adjoint{T,Vector{T}}) where T
    x = parent(data)
    @test Tensar(data) == Tensar{eltype(x),0,1}(conj(x))
    @test Array(data) == x'
end

function implicit_shape(data::AbstractArray)
    @test Tensar(data) ==
        (ndims(data) == 2) ?
        Tensar(data,1,1) :
        Tensar(data,ndims(data),0)
end
