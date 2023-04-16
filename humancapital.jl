# Build a model of (stochastic) human capital accumulation
# All wages consumed every period (no saving dynamic)
# Working more hours brings disutility, but increases wages in current period
# and adds to human capital levels
# More human capital increases wages
# Later add option to work 0 hours

using Interpolations, Optim, Distributions
using Optim: maximizer

@show Threads.nthreads()

# Parameters
T = 20
β = 0.9
k₁= 100
n_k_grid = 1000
h_min = 0
h_max = 60
global const A = 0.1    # Constant multiplier to calibrate wage schedule
global const ρ = 0.6    # The elasticity of wages with respect to total accumulated human capital
global const θ = 0.5    # The elasticity of wages with respect to total hours worked this period
global const γ = 0.33

# Every period, a shock changes the accumulated human capital by a proportion
σ = 0.1
shock_dist = LogNormal(0, σ)
# Discrete the shock by using the quantile function
# and uniformly sampling from the discretized quantile function
n_ϵ_grid = 20
ϵ_grid = [quantile(shock_dist, i/(1+n_ϵ_grid)) for i in 1:n_ϵ_grid]
ϵ_max = ϵ_grid[n_ϵ_grid]

k_min = 0  # Minimum Human Capital level - but we'll never hit this anyway
k_max = ϵ_max^(T-1)*k₁ + h_max * sum([(ϵ_max^(t)) for t in 1:T-1])

V = zeros(T+1, n_k_grid) # V[t,i] is Vₜ(kⁱ) - the value function if you start period t with human capital of kⁱ
policy = zeros(T+1, n_k_grid) # policy[t,i] is the optimal hours worked if at period t you have human capital kⁱ
k_grid = collect(range(k_min, stop=k_max, length=n_k_grid)) # Grid of state variables (human capital)

wage(kₜ,hₜ) = A * kₜ^ρ * hₜ^θ
transition(kₜ, hₜ, ϵₜ) = (kₜ + hₜ)*ϵₜ
pmf(ϵₜ) = 1/n_ϵ_grid # The odds of drawing any discrete shock is constant, because that's how we discretized/transformed it
u(kₜ,hₜ) = log(wage(kₜ,hₜ)) - (hₜ^(1+γ))/(1+γ) # Stage game payoff - utility from wages, disutility from hours worked

function make_Ṽ(V, k_grid, t)
    interpolate((k_grid,), V[t, :], Gridded(Linear()))
end

# Eₜ(Ṽₜ₊₁(hₜ₊₁)|hₜ, kₜ)
# Expected interpolated value function in the next period given your current state and action
# As the state transition is stochastic
function Eₜ(Ṽₜ₊₁, hₜ, kₜ)
    # Check the interpolated value function for every possible realization of the noise
    # and as such every possible realization of hₜ given the current state and action
    # average the results
    mean(map(ϵₜ -> Ṽₜ₊₁(transition(kₜ, hₜ, ϵₜ)), ϵ_grid))
end

# The maximand in every period - the stage utility plus the expected value function of the next period
# given our actions this period and current state
function maximand(Ṽₜ₊₁, kₜ)
    function(hₜ)
        u(kₜ, hₜ) + Eₜ(Ṽₜ₊₁, hₜ, kₜ)
    end
end

# Maximizes the stage game of the DP problem
function maximize_stage(Ṽₜ₊₁, kₜ, h_min, h_max)
    f = maximand(Ṽₜ₊₁, kₜ)
    maximize(f, h_min, h_max)
end




