using ControlSystemIdentification, ControlSystems
Random.seed!(1)
T = 1000
nx = 3
nu = 1
ny = 1
x0 = randn(nx)
sim(sys, u, x0 = x0) = lsim(sys, u', 1:T, x0 = x0)[1]'
sim(sys::KalmanFilter, u) = simulate(sys,u)

sys = c2d(ControlSystems.DemoSystems.resonant() * tf(1, [0.1, 1]), 1)# generate_system(nx, nu, ny)
sysn = generate_system(nx, nu, ny)

σu = 1e-3
σy = 1e-3


error("höll på med simulate(kf, u), var fel i typerna från similar")
# NOTE: SEM through increasing meas noise cov does not work, the feedback from meas decreases, but the loglik does not get penalized by poor predictoin performance. If one could decrease feedback while maintaining the same cov in the loglik calculation it would perhaps work. I tried using actual simulation error but didn't finish it, that would be another strategy.

u  = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d  = iddata(yn, un, 1)

# using BenchmarkTools
# @btime begin
# Random.seed!(0)
sys0 = n4sid(d, nx)
sysh, kf, opt = pem(d, nx = nx, focus = :prediction, iterations=500, show_trace=true, show_every=50, sys0=sys0, x_tol=1e-6)
x0h = kf.d0.μ
# bodeplot([sys,ss(sysh)], exp10.(range(-3, stop=log10(pi), length=150)), legend=false, ylims=(0.01,100))
# end
# 462ms 121 29
# 296ms
# 283ms
# 173ms
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test freqresptest(sys, sysh) < 1e-1
yh = sim(sysh, u)

# plot([vec(y) vec(yh)])

# Test with some noise
# Only measurement noise
σu = 0.0
σy = 0.1
u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
sys0 = n4sid(d, nx)
sysh, kf, opt = pem(d, nx = nx, focus = :prediction, show_trace=true, show_every=50, sys0=sys0, x_tol=1e-4)
x0h = kf.d0.μ
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test freqresptest(sys, sysh) < 0.5
# bodeplot([sys,ss(sysh)], exp10.(range(-3, stop=log10(pi), length=150)), legend=false, ylims=(0.01,100))

# Only input noise
σu = 0.1
σy = 0.0
u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
sys0 = n4sid(d, nx)
@time sysh, kf, opt = pem(d, nx = nx, focus = :prediction, show_trace=true, show_every=50, sys0=sys0, x_tol=1e-6)
x0h = kf.d0.μ
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test freqresptest(sys, sysh) < 1e-2
# bodeplot([sys,sys0.sys, ss(sysh)], exp10.(range(-3, stop=log10(pi), length=150)), legend=false, ylims=(0.01,100))

# Both noises
σu = 0.2
σy = 0.2

u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
sys0 = nothing#n4sid(d, nx)
sysh, kf, opt = pem(d, nx = 2nx, focus = :prediction, iterations = 400, show_trace=true, show_every=20, solver=BFGS(), sys0=sys0, x_tol=1e-6)
x0h = kf.d0.μ
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test freqresptest(sys, sysh) < 1e-2

# Simulation error minimization
σu = 0.01
σy = 0.01

u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
@time sysh, kf, opt = pem(d, nx = nx, focus = :simulation, show_trace=true, show_every=50, sys0=sys0, x_tol=1e-4, iterations=20)
x0h = kf.d0.μ
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.3
@test freqresptest(sys, sysh) < 1e-2

# plot([vec(y) vec(sim(sysh, u))])