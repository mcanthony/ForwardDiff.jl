[![Build Status](https://travis-ci.org/JuliaDiff/ForwardDiff.jl.svg?branch=nduals-refactor)](https://travis-ci.org/JuliaDiff/ForwardDiff.jl) [![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=nduals-refactor&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=nduals-refactor)

# ForwardDiff.jl

The `ForwardDiff` package provides a type-based implementation of forward mode automatic differentiation (FAD) in Julia. [The wikipedia page on automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is a useful resource for learning about the advantages of FAD techniques over other common differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

## What can I do with this package?

This package contains methods to efficiently take derivatives, Jacobians, and Hessians of native Julia functions (or any callable object, really). While performance varies depending on the functions you evaluate, this package generally outperforms non-AD methods in memory usage, speed, and accuracy.

A third-order generalization of the Hessian is also implemented (see `tensor` below). 

For now, we only support for functions involving `T<:Real`s, but we believe extension to numbers of type `T<:Complex` is possible.

## Usage

---
#### Derivative of `f: R → R` or `f: R → Rᵐ¹ × Rᵐ² × ⋯ × Rᵐⁱ`
---

- **`derivative!(output::Array, f, x::Number)`**
    
    Compute `f'(x)`, storing the output in `output`.

- **`derivative(f, x::Number)`**
    
    Compute `f'(x)`.

- **`derivative(f; mutates=false)`**
    
    Return the function `f'`. If `mutates=false`, then the returned function has the form `derivf(x) -> derivative(f, x)`. If `mutates = true`, then the returned function has the form `derivf!(output, x) -> derivative!(output, f, x)`.

---
#### Gradient of `f: Rⁿ → R`
---

- **`gradient!(output::Vector, f, x::Vector)`**

    Compute `∇f(x)`, storing the output in `output`.

- **`gradient{T}(f, x::Vector{T})`**

    Compute `∇f(x)`, where `T` is the element type of both the input and output.

- **`gradient(f; mutates=false)`**

    Return the function `∇f`. If `mutates=false`, then the returned function has the form `gradf(x) -> gradient(f, x)`. If `mutates = true`, then the returned function has the form `gradf!(output, x) -> gradient!(output, f, x)`. By default, `mutates` is set to `false`.

---
#### Jacobian of `f: Rⁿ → Rᵐ`
---

- **`jacobian!(output::Matrix, f, x::Vector)`**

    Compute `J(f(x))`, storing the output in `output`.

- **`jacobian{T}(f, x::Vector{T})`**

    Compute `J(f(x))`, where `T` is the element type of both the input and output.

- **`jacobian(f; mutates=false)`**

    Return the function `J(f)`. If `mutates=false`, then the returned function has the form `jacf(x) -> jacobian(f, x)`. If `mutates = true`, then the returned function has the form `jacf!(output, x) -> jacobian!(output, f, x)`. By default, `mutates` is set to `false`.

---
#### Hessian of `f: Rⁿ → R`
---

- **`hessian!(output::Matrix, f, x::Vector)`**

    Compute `H(f(x))`, storing the output in `output`.

- **`hessian{T}(f, x::Vector{T})`**

    Compute `H(f(x))`, where `T` is the element type of both the input and output.

- **`hessian(f; mutates=false)`**

    Return the function `H(f)`. If `mutates=false`, then the returned function has the form `hessf(x) -> hessian(f, x, S)`. If `mutates = true`, then the returned function has the form `hessf!(output, x) -> hessian!(output, f, x)`. By default, `mutates` is set to `false`.

---
#### Third-order Taylor series term of `f: Rⁿ → R`
---

[This Math StackExchange post](http://math.stackexchange.com/questions/556951/third-order-term-in-taylor-series) actually has an answer that explains this term fairly clearly.

- **`tensor!{S}(output::Array{S,3}, f, x::Vector)`**

    Compute `∑D³f(x)`, storing the output in `output`.

- **`tensor{T}(f, x::Vector{T})`**

    Compute `∑D³f(x)`, where `T` is the element type of both the input and output.

- **`tensor(f; mutates=false)`**

    Return the function ``∑D³f``. If `mutates=false`, then the returned function has the form `tensf(x) -> tensor(f, x)`. If `mutates = true`, then the returned function has the form `tensf!(output, x) -> tensor!(output, f, x)`. By default, `mutates` is set to `false`.

## Taking derivatives/gradients/etc. w.r.t specific function arguments

All of the above functions support evaluation of the objective function with respect to a specific argument. To do so, first pass in `wrt{i}` (where `i` is the index of the target argument), and then pass in the rest of the arguments as usual:

```julia

julia> f(x,y,z) = x^2 * sin(y) + 2*y*x + z^2
f (generic function with 1 method)

julia> derivative(wrt{1}, f, 1.0, 1.0, 1.0) # derivative of δf/δx at (1.0,1.0,1.0)
3.682941969615793

julia> derivative(wrt{2}, f, 1.0, 1.0, 1.0) # derivative of δf/δy at (1.0,1.0,1.0)
2.5403023058681398

julia> derivative(wrt{3}, f, 1.0, 1.0, 1.0) # derivative of δf/δz at (1.0,1.0,1.0)
2.0
```

This works for the mutating versions of the functions as well:

```julia

julia> f(x,y,z) = [x^2 * sin(y), 2*y*x, z^2]
f (generic function with 1 method)

julia> derivative!(wrt{3}, Vector{Float64}(3), f, 1.0, 1.0, 1.0) # derivative of δf/δz at (1.0,1.0,1.0)
3-element Array{Float64,1}:
 0.0
 0.0
 2.0

```

...as well as on closures returned by function calls like `derivative(f)`:

```julia
julia> derivf! = derivative(f, mutates=true)
derivf! (generic function with 2 methods)

julia> derivf!(wrt{3}, Vector{Float64}(3), 1.0, 1.0, 1.0)
3-element Array{Float64,1}:
 0.0
 0.0
 2.0
```

Your objective function can even have arguments of mixed type, as long as the target argument is typed properly for the FAD method you're using:

```julia

julia> f(a::Number, b::Vector) = a * b[1] + b[2]*b[1]^a + b[3]^b[2]/a
f (generic function with 2 methods)

julia> hessian(wrt{2}, f, 3.0, [1.0, 2.0, 3.0]) # take the Hessian of f w.r.t b
3x3 Array{Float64,2}:
 12.0  3.0      0.0
  3.0  3.62085  3.19722
  0.0  3.19722  0.666667

```

Thus, if you're using `derivative`, then `wrt{i}` must point to an argument of type `Number`, while using `gradient`/`jacobian`/`hessian`/`tensor` means `wrt{i}` refers to an argument of type `Vector`.
