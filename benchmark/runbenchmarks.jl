# This file uses BenchmarkTrackers.jl: https://github.com/JuliaCI/BenchmarkTrackers.jl

using BenchmarkTrackers
using ForwardDiff

###########################
# Benchmarkable Functions #
###########################

function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

function ackley(x)
    a, b, c = 20.0, -0.2, 2.0*π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c*i)
        sum_sqrs += i^2
    end
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + e)
end

self_weighted_logit(x) = inv(1.0 + exp(-dot(x, x)))

benchgrad(f, x, chunk) = ForwardDiff.gradient(f, x, chunk_size=chunk)
benchhess(f, x, chunk) = ForwardDiff.hessian(f, x, chunk_size=chunk)

###########################
# Benchmarkable Functions #
###########################

tracker = BenchmarkTracker()

# We use the same benchmarks for all metadata blocks, just with different
# settings. Thus, we can just put the actual benchmarks settings in a variable
# for later reuse.
benches = @benchmarks begin
              ["gradient[$f,$(length(x)),$c]" => benchgrad(f, x, c) for f in funcs, x in xs, c in chunks]
              ["hessian[$f,$(length(x)),$c]"  => benchhess(f, x, c) for f in funcs, x in xs, c in chunks]
          end

@track tracker begin
    @setup begin
        funcs = (ackley, rosenbrock, self_weighted_logit)
        xs = (rand(16), rand(32))
        chunks = (ForwardDiff.default_chunk_size,1,16)
    end

    benches

    @constraints seconds=>10
    @tags "quick"
end

@track tracker begin
    @setup begin
        funcs = (ackley, rosenbrock, self_weighted_logit)
        xs = (rand(160), rand(1600))
        chunks = (ForwardDiff.default_chunk_size,1,2,4,8,16)
    end

    benches

    @tags "slow"
end

@declare_ci tracker
