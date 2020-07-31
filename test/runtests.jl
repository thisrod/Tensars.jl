using Test
using Tensars
using Tensars: RowVector, TensArray
using LinearAlgebra

randc(ns...) = randc(ns)
randc(ns::Tuple) = (randn(ns) .+ 1im*randn(ns))/√2

randv() = randc()
randv(::Tuple{}) = randc()
randv(ns::Tuple) = Tensar(randc(ns), length(ns), 0)
randv(ns...) = Tensar(randc(ns...), length(ns), 0)

randt(cs, rs) = reshape(randc(cs..., rs...), cs, rs)

A = randc(2,3,4,5,6)
B = randc(4,5,6,7,8)
M = randc(7,8)
x = randc(8)
notx = randc(8)
y = randc(7)
z = randc()
@assert x != notx

@testset "Tensar construction and equality" begin
    @test Tensar(z) == z
    @test Tensar(z,0,0) == z
    @test Tensar(M) == Tensar{eltype(M),1,1}(M)
    @test Tensar(x) == Tensar{eltype(x),1,0}(x)
    @test Tensar(x) != Tensar(notx)
    @test Tensar(x') == Tensar{eltype(x),0,1}(conj.(x))
    @test Array(Tensar(x')) == x'
    @test Tensar(reshape([2], ())) == 2
    @test Tensar(A) == Tensar{eltype(A),5,0}(A)
    @test Tensar(A,(2,3)) == Tensar{eltype(A),2,3}(A)
    @test Tensar(A,2,3) == Tensar(A,(2,3))
    @test_throws Exception Tensar(A, (2,3), (4,5,6))
    @test_throws DimensionMismatch Tensar{eltype(x),1,1}(x)
    @test_throws DimensionMismatch Tensar{eltype(x),0,0}(x)
    @test_throws DimensionMismatch Tensar{eltype(x),-1,0}(x)
    @test_throws DimensionMismatch Tensar{eltype(x),0,2//3}(x)
end

TA = Tensar(A,2,3)
TB = Tensar(B,3,2)
TC = Tensar(rand(2,3,4,5,6,7),3,3)
TM = Tensar(M)
TN = Tensar(randc(size(M)))
Tx = Tensar(x)
TP = Tensar(randc(7,7))
ket = Tensar(M,2,0)
bra = Tensar(conj.(M),0,2)

@testset "RowVector type" begin
    @test x' isa RowVector
    @test transpose(x) isa RowVector
    @test !(x isa RowVector)
    @test !(M' isa RowVector)
    @test_skip !(Tx' isa RowVector)
    @test_skip !(TM' isa RowVector)
    @test_skip !(TA' isa RowVector)
end

@testset "Rank, length and size" begin
    function test_correct_weight(A::Tensar{T,M,N}) where T where M where N
        @test eltype(A) == T
        @test ncols(A) == M
        @test nrows(A) == N
        @test ndims(A) == (M,N)
        @test length(colsize(A)) == M
        @test length(rowsize(A)) == N
        for j = 1:ncols(A)
            @test colsize(A,j) == colsize(A)[j]
        end
        for j = 1:nrows(A)
            @test rowsize(A,j) == rowsize(A)[j]
        end
        @test size(A) == (colsize(A), rowsize(A))
        @test length(A) == prod.(size(A))
    end
    test_correct_weight(TA)
    test_correct_weight(TM)
    test_correct_weight(Tensar(M, ndims(M), 0))
    test_correct_weight(Tensar(M, 0, ndims(M)))
    test_correct_weight(Tx)
    test_correct_weight(Tx')
end

@testset "Display" begin
    @test repr(TA) == "2×3 ← 4×5×6 Tensar{Complex{Float64}}"
    @test repr(Tx) == "8-vector ← scalar Tensar{Complex{Float64}}"
    @test repr(Tensar(x')) == "scalar ← 8-vector Tensar{Complex{Float64}}"
    @test repr(Tx') == "scalar ← 8-vector Tensar{Complex{Float64}}"
end

@testset "Zeros" begin
    @test zero(TA) == Complex(0.0)*TA
    @test zeros(eltype(TA), size(TA)...) == Complex(0.0)*TA
end

@testset "Reshape" begin
    @test reshape(A, (2,3), (4,5,6)) == TA
    @test reshape(A[:], (2,3), (4,5,6)) == TA
    # not yet implemented
    @test_skip reshape(TA,2,3,4,5,6) == A
end

@testset "Indexing" begin
    @test size(TA[2,:,:,3,:]) == ((3,), (4,6))
    @test size(TA[:,1:2,:,:,:]) == ((2,2), (4,5,6))
end

@testset "Permutedims" begin
    # not yet implemented
end

@testset "Addition" begin
    @test TA + TA ≈ Complex(2.0)*TA
    @test TM + TM ≈ Complex(2.0)*TM
    @test_skip TM' + TM' ≈ Complex(2.0)*TM'
    @test Tx + Tx ≈ Complex(2.0)*Tx
    @test Tx' + Tx' ≈ Complex(2.0)*Tx'
end

@testset "Product return types" begin
    @test bra*Array(ket) isa Complex
    @test bra*ket isa Complex
    @test y'*TM isa RowVector
    @test TB*M isa Array
    @test TA*TB isa Tensar
    @test Complex(2.0)*TA isa typeof(TA)
    @test TA*Complex(2.0) isa typeof(TA)
    @test size(Complex(2.0)*TA) == size(TA)
    @test size(TA*Complex(2.0)) == size(TA)
end

@testset "Product values" begin
    @test bra*ket ≈ bra*Array(ket)
    @test Tx'*Tx ≈ dot(x,x)
    @test TM*x ≈ M*x
    @test y'*TM ≈ y'*M
    @test (TA*TB)*M ≈ TA*(TB*M)
    @test Array(Complex(2.0)*TA) == 2*Array(A)
    @test Array(TA*Complex(2.0)) == 2*Array(A)
end

@testset "Partial contraction" begin
    A = randt((3,), (4,5))
    x = randc(4,5,6)
    @test size(A∗x) == (3,6)
    Ax = similar(x,3,6)
    for j = 1:6
        Ax[:,j] = A*x[:,:,j]
    end
    @test A∗x ≈ Ax
end

@testset "Adjoints" begin
    # TODO set up correct_adjoint akin to correct_outer_product
    @test bra ≈ ket'
    @test Tx' == Tensar(x')
    @test Tensar(M,0,2)' == Tensar(conj.(M),2,0)
    @test Tensar(M,2,0)' == Tensar(conj.(M),0,2)
    @test (Tx')' == Tx
end

args_for(z::Complex) = randc(2)
args_for(A::Tensar) = (randv(colsize(A))', randv(rowsize(A)))

function correct_outer_product(A,B)
    # treat the tensors as multilinear forms
    dA, vA = args_for(A)
    dB, vB = args_for(B)
    (dA⊗dB)*(A⊗B)*(vA⊗vB) ≈ dA*A*vA*dB*B*vB
end

@testset "Outer products" begin
    @test correct_outer_product(randc(), randc())
    @test correct_outer_product(randc(), TM)
    @test correct_outer_product(TM, randc())
    @test correct_outer_product(randv(3), randv(4))
    @test correct_outer_product(randv(3)', randv(4))
    @test correct_outer_product(randv(3), randv(4)')
    @test correct_outer_product(randv(3)', randv(4)')
    @test correct_outer_product(TM, randv(3))
    @test correct_outer_product(TM, randv(3)')
    @test correct_outer_product(randv(3), TM)
    @test correct_outer_product(randv(3)', TM)
    @test correct_outer_product(TM, TN)
    @test correct_outer_product(TA, TM)
end

@testset "Shape mismatch throws error" begin

end

@testset "Contraction" begin
    @test tr(Array(TP)) == tr(TP,1,1)
    @test tr(ket⊗TC,1,3) isa Tensar
end

@testset "Eltype promotion" begin
    # Not yet implemented
end

@testset "TensArray broadcasting" begin
    x = TensArray(rand(3,4),(3,),(4,))
    y = TensArray(rand(3,4), (3,), (2,2))
    
    @test x .+ x isa TensArray
    @test x .+ rand() isa TensArray
    @test x .+ rand() .* x isa TensArray
    @test rand() .+ x isa TensArray
    @test x .+ y isa Array
    @test y .+ x isa Array
    @test x .+ rand(3,4) isa Array
    @test rand(3,4) .+ x isa Array
    @test x .+ rand(3) isa Array
    @test rand(3) .+ x isa Array
    @test x .+ rand(1,4) isa Array
    @test rand(1,4) .+ x isa Array
end

function commutes_with_cast(op, args...)
    op(args...) ≈ Tensar(op(TensArray.(args)...))
end

@testset "TensArray and Tensar casts" begin
    TAA = TensArray(TA)
    @test TAA isa TensArray
    @test size(TAA) == length(TA)
    @test Tensar(TAA) == TA
    @test commutes_with_cast(+, TA, TA)
    @test commutes_with_cast(+, TM, TM)
    @test_skip commutes_with_cast(+, TM', TM')
    @test commutes_with_cast(+, Tx, Tx)
    @test commutes_with_cast(+, Tx', Tx')
    @test commutes_with_cast(*, bra, ket)
    @test commutes_with_cast(*, Tx', Tx)
    @test Tensar((TensArray(TA)*TensArray(TB))*TensArray(ket)) ≈ TA*(TB*ket)
end