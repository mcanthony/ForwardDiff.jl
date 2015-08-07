abstract wrt{i}
abstract dim{i}

# resolve ambiguity warnings
derivative{i}(::Type{wrt{i}}, x::Number) = error("derivative has no method derivative(::Type{wrt{i}}, x::Number)")
gradient{i}(::Type{wrt{i}}, x::Vector) = error("gradient has no method gradient(::Type{wrt{i}}, x::Vector)")
jacobian{i}(::Type{wrt{i}}, x::Vector) = error("jacobian has no method jacobian(::Type{wrt{i}}, x::Vector)")
hessian{i}(::Type{wrt{i}}, x::Vector) = error("hessian has no method hessian(::Type{wrt{i}}, x::Vector)")
tensor{i}(::Type{wrt{i}}, x::Vector) = error("tensor has no method tensor(::Type{wrt{i}}, x::Vector)")

######################
# Taking Derivatives #
######################

# Load derivative from ForwardDiffNum #
#-------------------------------------#
function load_derivative!(output::Array, arr::Array)
    @assert length(arr) == length(output)
    @simd for i in eachindex(output)
        @inbounds output[i] = grad(arr[i], 1)
    end
    return output
end

load_derivative(arr::Array) = load_derivative!(similar(arr, eltype(eltype(arr))), arr)
load_derivative(n::ForwardDiffNum{1}) = grad(n, 1)

# Derivative from function/Exposed API methods #
#----------------------------------------------#

@generated function calc_derivnum{N,i}(f, args::NTuple{N}, ::Type{wrt{i}})
    f_args = wrt_args(N, :g, i)
    return quote 
        x = args[$i]
        g = GradientNum(x, one(x))
        return f($(f_args.args...))
    end
end

derivative!{i}(Wrt::Type{wrt{i}}, output::Array, f, args...) = load_derivative!(output, calc_derivnum(f, args, Wrt))
derivative!(output::Array, f, x::Number) = derivative!(wrt{1}, output, f, x)

derivative{i}(Wrt::Type{wrt{i}}, f, args...) = load_derivative(calc_derivnum(f, args, Wrt))
derivative(f, x::Number) = derivative(wrt{1}, f, x)

function derivative(f; mutates::Bool=false)
    if mutates
        derivf!(output::Array, x::Number) = derivative!(output, f, x)
        derivf!{i}(Wrt::Type{wrt{i}}, output::Array, args...) = derivative!(Wrt, output, f, args...)
        return derivf!
    else
        derivf(x::Number) = derivative(f, x)
        derivf{i}(Wrt::Type{wrt{i}}, args...) = derivative(Wrt, f, args...)
        return derivf
    end
end

####################
# Taking Gradients #
####################

# Load gradient from ForwardDiffNum #
#-----------------------------------#
function load_gradient!(output::Vector, n::ForwardDiffNum)
    @assert npartials(n) == length(output) "The output vector must be the same length as the input vector"
    @simd for i in eachindex(output)
        @inbounds output[i] = grad(n, i)
    end
    return output
end

load_gradient{N,T,C}(n::ForwardDiffNum{N,T,C}) = load_gradient!(Array(T, N), n)

# Gradient from function #
#------------------------#
function load_gradvec!{N,T,C}(gradvec::Vector{GradientNum{N,T,C}}, x::Vector{T})
    G = eltype(gradvec)
    
    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(gradvec) == N "The GradientNum vector must be the same length as the input vector"

    pchunk = partials_chunk(G)

    @simd for i in eachindex(gradvec)
        @inbounds gradvec[i] = G(x[i], pchunk[i])
    end

    return gradvec
end

function take_gradient!{i}(output::Vector, f, args::Tuple, Wrt::Type{wrt{i}}, gradvec::Vector)
    return load_gradient!(output, calc_fadnum!(f, args, Wrt, gradvec, load_gradvec!))
end

function take_gradient!{i,d}(output::Vector, f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_gradient!(output, f, args, Wrt, grad_workvec(Dim, eltype(args[i])))
end

function take_gradient!{i}(f, args::Tuple, Wrt::Type{wrt{i}}, gradvec::Vector)
    return load_gradient(calc_fadnum!(f, args, Wrt, gradvec, load_gradvec!))
end

function take_gradient{i,d}(f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_gradient!(f, args, Wrt, grad_workvec(Dim, eltype(args[i])))
end

# Exposed API methods #
#---------------------#
function gradient!{T,i}(Wrt::Type{wrt{i}}, output::Vector{T}, f, args...)
    return take_gradient!(output, f, args, Wrt, dim{length(args[i])})::Vector{T}
end

gradient!{T}(output::Vector{T}, f, x::Vector) = gradient!(wrt{1}, output, f, x)

function gradient{i}(Wrt::Type{wrt{i}}, f, args...)
    x = args[i]
    return take_gradient(f, args, Wrt, dim{length(x)})::Vector{eltype(x)}
end

gradient(f, x::Vector) = gradient(wrt{1}, f, x)

function gradient(f; mutates::Bool=false)
    if mutates
        gradf!(output::Vector, x::Vector) = gradient!(output, f, x)
        gradf!{i}(Wrt::Type{wrt{i}}, output::Vector, args...) = gradient!(Wrt, output, f, args...)
        return gradf!
    else
        gradf(x::Vector) = gradient(f, x)
        gradf{i}(Wrt::Type{wrt{i}}, args...) = gradient(Wrt, f, args...)
        return gradf
    end
end

####################
# Taking Jacobians #
####################

# Load Jacobian from ForwardDiffNum #
#-----------------------------------#
function load_jacobian!(output, jacvec::Vector)
    # assumes jacvec is actually homogenous,
    # though it may not be well-inferenced.
    N = npartials(first(jacvec))
    for i in 1:length(jacvec), j in 1:N
        output[i,j] = grad(jacvec[i], j)
    end
    return output
end

function load_jacobian(jacvec::Vector)
    # assumes jacvec is actually homogenous,
    # though it may not be well-inferenced.
    F = typeof(first(jacvec))
    return load_jacobian!(Array(eltype(F), length(jacvec), npartials(F)), jacvec)
end

# Jacobian from function #
#------------------------#
function take_jacobian!{i}(output::Matrix, f, args::Tuple, Wrt::Type{wrt{i}}, gradvec::Vector)
    return load_jacobian!(output, calc_fadnum!(f, args, Wrt, gradvec, load_gradvec!))
end

function take_jacobian!{i,d}(output::Matrix, f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_jacobian!(output, f, args, Wrt, grad_workvec(Dim, eltype(args[i])))
end

function take_jacobian!{i}(f, args::Tuple, Wrt::Type{wrt{i}}, gradvec::Vector)
    return load_jacobian(calc_fadnum!(f, args, Wrt, gradvec, load_gradvec!))
end

function take_jacobian{i,d}(f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_jacobian!(f, args, Wrt, grad_workvec(Dim, eltype(args[i])))
end

# Exposed API methods #
#---------------------#
function jacobian!{T,i}(Wrt::Type{wrt{i}}, output::Matrix{T}, f, args...)
    return take_jacobian!(output, f, args, Wrt, dim{length(args[i])})::Matrix{T}
end

jacobian!{T}(output::Matrix{T}, f, x::Vector) = jacobian!(wrt{1}, output, f, x)

function jacobian{i}(Wrt::Type{wrt{i}}, f, args...)
    x = args[i]
    return take_jacobian(f, args, Wrt, dim{length(x)})::Matrix{eltype(x)}
end

jacobian(f, x::Vector) = jacobian(wrt{1}, f, x)

function jacobian(f; mutates::Bool=false)
    if mutates
        jacf!(output::Matrix, x::Vector) = jacobian!(output, f, x)
        jacf!{i}(Wrt::Type{wrt{i}}, output::Matrix, args...) = jacobian!(Wrt, output, f, args...)
        return jacf!
    else
        jacf(x::Vector) = jacobian(f, x)
        jacf{i}(Wrt::Type{wrt{i}}, args...) = jacobian(Wrt, f, args...)
        return jacf
    end
end

###################
# Taking Hessians #
###################

# Load Hessian from ForwardDiffNum #
#----------------------------------#
function load_hessian!{N}(output, n::ForwardDiffNum{N})
    @assert (N, N) == size(output) "The output matrix must have size (length(input), length(input))"
    q = 1
    for i in 1:N
        for j in 1:i
            val = hess(n, q)
            @inbounds output[i, j] = val
            @inbounds output[j, i] = val
            q += 1
        end
    end
    return output
end

load_hessian{N,T}(n::ForwardDiffNum{N,T}) = load_hessian!(Array(T, N, N), n)

# Hessian from function #
#-----------------------#
function load_hessvec!{N,T,C}(hessvec::Vector{HessianNum{N,T,C}}, x::Vector{T}) 
    G = GradientNum{N,T,C}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(hessvec) == N "The HessianNum vector must be the same length as the input vector"

    pchunk = partials_chunk(G)
    zhess = zero_partials(eltype(hessvec))

    @simd for i in eachindex(hessvec)
        @inbounds hessvec[i] = HessianNum(G(x[i], pchunk[i]), zhess)
    end

    return hessvec
end

function take_hessian!{i}(output::Matrix, f, args::Tuple, Wrt::Type{wrt{i}}, hessvec::Vector)
    return load_hessian!(output, calc_fadnum!(f, args, Wrt, hessvec, load_hessvec!))
end

function take_hessian!{i,d}(output::Matrix, f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_hessian!(output, f, args, Wrt, hess_workvec(Dim, eltype(args[i])))
end

function take_hessian!{i}(f, args::Tuple, Wrt::Type{wrt{i}}, hessvec::Vector)
    return load_hessian(calc_fadnum!(f, args, Wrt, hessvec, load_hessvec!))
end

function take_hessian{i,d}(f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_hessian!(f, args, Wrt, hess_workvec(Dim, eltype(args[i])))
end

# Exposed API methods #
#---------------------#
function hessian!{T,i}(Wrt::Type{wrt{i}}, output::Matrix{T}, f, args...)
    return take_hessian!(output, f, args, Wrt, dim{length(args[i])})::Matrix{T}
end

hessian!{T}(output::Matrix{T}, f, x::Vector) = hessian!(wrt{1}, output, f, x)

function hessian{i}(Wrt::Type{wrt{i}}, f, args...)
    x = args[i]
    return take_hessian(f, args, Wrt, dim{length(x)})::Matrix{eltype(x)}
end

hessian(f, x::Vector) = hessian(wrt{1}, f, x)

function hessian(f; mutates::Bool=false)
    if mutates
        hessf!(output::Matrix, x::Vector) = hessian!(output, f, x)
        hessf!{i}(Wrt::Type{wrt{i}}, output::Matrix, args...) = hessian!(Wrt, output, f, args...)
        return hessf!
    else
        hessf(x::Vector) = hessian(f, x)
        hessf{i}(Wrt::Type{wrt{i}}, args...) = hessian(Wrt, f, args...)
        return hessf
    end
end

##################
# Taking Tensors #
##################

# Load Tensor from ForwardDiffNum #
#---------------------------------#
function load_tensor!{N,T,C}(output, n::ForwardDiffNum{N,T,C})
    @assert (N, N, N) == size(output) "The output array must have size (length(input), length(input), length(input))"
    
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                @inbounds output[j, k, i] = tens(n, q)
                q += 1
            end
        end

        for j in 1:(i-1)
            for k in 1:j
                @inbounds output[j, k, i] = output[i, j, k]
            end
        end

        for j in i:N
            for k in 1:(i-1)
                @inbounds output[j, k, i] = output[i, j, k]
            end
        end

        for j in 1:N
            for k in (j+1):N
                @inbounds output[j, k, i] = output[k, j, i]
            end
        end
    end

    return output
end

load_tensor{N,T,C}(n::ForwardDiffNum{N,T,C}) = load_tensor!(Array(T, N, N, N), n)

# Tensor from function #
#----------------------#
function load_tensvec!{N,T,C}(tensvec::Vector{TensorNum{N,T,C}}, x::Vector{T}) 
    G = GradientNum{N,T,C}
    H = HessianNum{N,T,C}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(tensvec) == N "The TensorNum vector must be the same length as the input"

    pchunk = partials_chunk(G)
    zhess = zero_partials(H)
    ztens = zero_partials(eltype(tensvec))

    @simd for i in eachindex(tensvec)
        @inbounds tensvec[i] = TensorNum(H(G(x[i], pchunk[i]), zhess), ztens)
    end

    return tensvec
end

function take_tensor!{T,i}(output::Array{T,3}, f, args::Tuple, Wrt::Type{wrt{i}}, tensvec::Vector)
    return load_tensor!(output, calc_fadnum!(f, args, Wrt, tensvec, load_tensvec!))
end

function take_tensor!{T,i,d}(output::Array{T,3}, f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_tensor!(output, f, args, Wrt, tens_workvec(Dim, eltype(args[i])))
end

function take_tensor!{i}(f, args::Tuple, Wrt::Type{wrt{i}}, tensvec::Vector)
    return load_tensor(calc_fadnum!(f, args, Wrt, tensvec, load_tensvec!))
end

function take_tensor{i,d}(f, args::Tuple, Wrt::Type{wrt{i}}, Dim::Type{dim{d}})
    return take_tensor!(f, args, Wrt, tens_workvec(Dim, eltype(args[i])))
end

# Exposed API methods #
#---------------------#
function tensor!{T,i}(Wrt::Type{wrt{i}}, output::Array{T,3}, f, args...)
    return take_tensor!(output, f, args, Wrt, dim{length(args[i])})::Array{T,3}
end

tensor!{T}(output::Array{T,3}, f, x::Vector) = tensor!(wrt{1}, output, f, x)

function tensor{i}(Wrt::Type{wrt{i}}, f, args...)
    x = args[i]
    return take_tensor(f, args, Wrt, dim{length(x)})::Array{eltype(x),3}
end

tensor(f, x::Vector) = tensor(wrt{1}, f, x)

function tensor(f; mutates::Bool=false)
    if mutates
        tensf!{T}(output::Array{T,3}, x::Vector) = tensor!(output, f, x)
        tensf!{T,i}(Wrt::Type{wrt{i}}, output::Array{T,3}, args...) = tensor!(Wrt, output, f, args...)
        return tensf!
    else
        tensf(x::Vector) = tensor(f, x)
        tensf{i}(Wrt::Type{wrt{i}}, args...) = tensor(Wrt, f, args...)
        return tensf
    end
end

####################
# Helper Functions #
####################
# Use @generated functions to essentially cache the
# zeros/partial components generated by the input type.
# This caching allows for a higher degree of efficieny 
# when calculating derivatives of f at multiple points, 
# as these values get reused rather than instantiated 
# every time.
#
# This method has the potential to incur a large memory 
# cost (and could even be considered a leak) if a 
# downstream program use many *different* partial components, 
# though I can't think of any use cases in which that would be 
# relevant.

# Load the input vector into a ForwardDiffNum vector using 
# load_func!, then pass the loaded work vector + args through f.
@generated function calc_fadnum!{N,i}(f, 
                                        args::NTuple{N}, 
                                        ::Type{wrt{i}}, 
                                        workvec::Vector, 
                                        load_func!::Function)
    f_args = wrt_args(N, :workvec, i)
    return quote 
        x = args[$i]
        workvec = load_func!(workvec, x)
        return f($(f_args.args...))
    end
end

function wrt_args(N::Int, item::Symbol, wrt::Int)
    return Expr(:tuple, [i == wrt ? item : :(args[$i]) for i=1:N]...)
end

@generated function pick_implementation{N,T}(::Type{dim{N}}, ::Type{T})
    if N > 10
        return :(Vector{$T})
    else
        return :(NTuple{$N,$T})
    end
end

@generated function grad_workvec{N,T}(::Type{dim{N}}, ::Type{T})
    result = Vector{GradientNum{N,T,pick_implementation(dim{N},T)}}(N)
    return :($result)
end

@generated function hess_workvec{N,T}(::Type{dim{N}}, ::Type{T})
    result = Vector{HessianNum{N,T,pick_implementation(dim{N},T)}}(N)
    return :($result)
end

@generated function tens_workvec{N,T}(::Type{dim{N}}, ::Type{T})
    result = Vector{TensorNum{N,T,pick_implementation(dim{N},T)}}(N)
    return :($result)
end

@generated function zero_partials{N,T}(::Type{GradNumVec{N,T}})
    result = zeros(T, N)
    return :($result)
end

@generated function zero_partials{N,T}(::Type{GradNumTup{N,T}})
    z = zero(T)
    result = ntuple(i->z, Val{N})
    return :($result)
end

@generated function zero_partials{N,T,C}(::Type{HessianNum{N,T,C}})
    result = zeros(T, halfhesslen(N))
    return :($result)
end

@generated function zero_partials{N,T,C}(::Type{TensorNum{N,T,C}})
    result = zeros(T, halftenslen(N))
    return :($result) 
end

@generated function partials_chunk{N,T}(::Type{GradNumVec{N,T}})
    dus_arr = Array(Vector{T}, N)
    @simd for i in eachindex(dus_arr)
        @inbounds dus_arr[i] = setindex!(zeros(T,N), one(T), i)
    end
    return :($dus_arr)
end

@generated function partials_chunk{N,T}(::Type{GradNumTup{N,T}})
    dus_arr = Array(NTuple{N,T}, N)
    z = zero(T)
    o = one(T)
    @simd for i in eachindex(dus_arr)
        @inbounds dus_arr[i] = ntuple(x -> ifelse(x == i, o, z), Val{N})
    end
    return :($dus_arr)
end