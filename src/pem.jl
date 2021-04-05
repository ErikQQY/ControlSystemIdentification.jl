ControlSystems.StateSpace(kf::KalmanFilter, h) = ss(kf.A,kf.B,kf.C,0,h)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
"""
    sys, x0, opt = pem(data; nx, kwargs...)

System identification using the prediction-error method.

# Arguments:
- `data`: iddata object containing `y` and `u`.
    - `y`: Measurements, either a matrix with time along dim 2, or a vector of vectors
    - `u`: Control signals, same structure as `y`
- `nx`: Number of poles in the estimated system. Thus number should be chosen as number of system poles plus number of poles in noise models for measurement noise and load disturbances.
- `focus`: Either `:prediction` or `:simulation`. If `:simulation` is chosen, a two stage problem is solved with prediction focus first, followed by a refinement for simulation focus.
- `metric`: A Function determining how the size of the residuals is measured, default `sse` (e'e), but any Function such as `norm`, `e->sum(abs,e)` or `e -> e'Q*e` could be used.
- `regularizer(p)=0`: function for regularization. The structure of `p` is detailed below
- `solver` Defaults to `Optim.BFGS()`
- `stabilize_predictor=true`: Modifies the estimated Kalman gain `K` in case `A-KC` is not stable by moving all unstable eigenvalues to the unit circle.
- `difficult=false`: If the identification problem appears to be difficult and ends up in a local minimum, set this flag to true to solve an initial global optimization problem to supply a good initial guess. This is expected to take some time.
- `kwargs`: additional keyword arguments are sent to `Optim.Options`.

# Return values
- `sys::StateSpaceNoise`: identified system. Can be converted to `StateSpace` by `convert(StateSpace, sys)`, but this will discard the Kalman gain matrix.
- `x0`: Estimated initial state
- `opt`: Optimization problem structure. Contains info of the result of the optimization problem

## Structure of parameter vector `p`
The parameter vector is of type [`ComponentVector`](https://github.com/jonniedie/ComponentArrays.jl) and the fields `A,B,K,x0` can be accessed as `p.A` etc. The internal storage is according to
```julia
A = size(nx,nx)
B = size(nx,nu)
K = size(nx,ny)
x0 = size(nx)
p = [A[:]; B[:]; K[:]; x0]
```
"""
function pem(
    d;
    nx,
    solver              = BFGS(),
    focus               = :prediction,
    regularizer::F      = p -> 0,
    iterations          = 1000,
    sys0                = nothing,
    kwargs...,
) where F

    y, u = tosvec(output(d)), tosvec(input(d))
    nu, ny = obslength(u), obslength(y)
    if sys0 === nothing
        sys0 = n4sid(d, nx)
    else
        nstates(sys0) == nx || throw(ArgumentError("Initial system sys0 does not have `nx` states."))
    end
    A,B,C = sys0.A, sys0.B, sys0.C
    p0 = ComponentVector((; A, B, C, x0=sys0.x[:,1]))
    options = Options(; iterations = iterations, kwargs...)
    function getsys(p, rm=1)
        T = eltype(p)
        A = SMatrix{nx,nx,T,nx^2}(p.A)
        Q = SMatrix{nx,nx,T,nx^2}(T.(sys0.Q + eps()*I))
        R = SMatrix{ny,ny,T,ny*ny}(T.(sys0.R + eps()*I)) * rm
        B = SMatrix{nx,nu,T,nx*nu}(p.B)
        C = SMatrix{ny,nx,T,nx*ny}(p.C)
        KalmanFilter(A,B,C, 0, Q, R, MvNormal(p.x0, Matrix(10Q)))
    end
    cost_pred = function (p)
        kf = getsys(p)
        try
            return -loglik(kf, u, y) + regularizer(p)
        catch
            return eltype(p)(Inf)
        end
    end

    opt = optimize(cost_pred, p0, solver, options; autodiff = :forward)
    println(opt)
    if focus == :simulation
        @info "Focusing on simulation"
        cost_sim = function (p)
            kf = getsys(p, 10000000)
            mean(LowLevelParticleFilters.simulate(kf,u) .- y) + regularizer(p)
        end
        opt = optimize(cost_sim, opt.minimizer, solver, options; autodiff = :forward)
        println(opt)
    end
    
    model = getsys(opt.minimizer)
    all(<=(1), abs.(eigvals(Matrix(model.A)))) || @warn("Estimated system is not stable")

    StateSpace(model, d.Ts), model, opt
end

function stabilize(model)
    s            = model.sys
    @unpack A, K = s
    C            = s.C
    poles        = eigvals(A - K * C)
    newpoles     = map(poles) do p
        ap = abs(p)
        ap <= 1 && (return p)
        p / (ap + sqrt(eps()))
    end
    K2           = ControlSystems.acker(A', C', newpoles)' .|> real
    all(abs(p) <= 1 for p in eigvals(A - K * C)) || @warn("Failed to stabilize predictor")
    s.K .= K2
    model
end
