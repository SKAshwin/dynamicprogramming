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
k₁= 100
n_k_grid = 10000
h_min = 0
h_max = 60
global const β = 0.9
global const A = 0.1    # Constant multiplier to calibrate wage schedule
global const ρ = 0.6    # The elasticity of wages with respect to total accumulated human capital
global const θ = 0.5    # The elasticity of wages with respect to total hours worked this period
global const γ = 0.33
global const ϕ = 0.01

# Every period, a shock changes the accumulated human capital by a proportion
σ = 0.1
shock_dist = LogNormal(0, σ)
# Discrete the shock by using the quantile function
# and uniformly sampling from the discretized quantile function
n_ϵ_grid = 20
ϵ_grid = [quantile(shock_dist, i/(1+n_ϵ_grid)) for i in 1:n_ϵ_grid]
ϵ_max = ϵ_grid[n_ϵ_grid]
ϵ_min = ϵ_grid[1]

function get_state_boundary(t, h_min, h_max, ϵ_min, ϵ_max, k₁)
    k_min = ϵ_min^(t-1)*k₁ + h_min * sum([(ϵ_max^(τ)) for τ in 1:t-1])
    k_max = ϵ_max^(t-1)*k₁ + h_max * sum([(ϵ_max^(τ)) for τ in 1:t-1])
    return (k_min, k_max)  
end

# Run a monte carlo simulation of outcomes in the state in stage T+1 if you play h_min every period
#function mc_min()

(grid_k_min, grid_k_max) = get_state_boundary(T+1, h_min, h_max, ϵ_min, ϵ_max, k₁) # Min/Max HC level at the start of period T+1
(k_min, k_max) = get_state_boundary(T, h_min, h_max, ϵ_min, ϵ_max, k₁) # Min/Max HC level at the start of period T

println("We have human capital space from $k_min to $k_max")

V = zeros(T+1, n_k_grid) # V[t,i] is Vₜ(kⁱ) - the value function if you start period t with human capital of kⁱ
policy = zeros(T+1, n_k_grid) # policy[t,i] is the optimal hours worked if at period t you have human capital kⁱ
k_grid = collect(range(grid_k_min, stop=grid_k_max, length=n_k_grid-2)) # Grid of state variables (human capital)
k_grid = sort(vcat(k_min, k_max, k_grid)) # make sure the actual stage T k_min, k_max are in the grid

wage(kₜ,hₜ) = A * kₜ^ρ * hₜ^θ
transition(kₜ, hₜ, ϵₜ) = (kₜ + hₜ)*ϵₜ
pmf(ϵₜ) = 1/n_ϵ_grid # The odds of drawing any discrete shock is constant, because that's how we discretized/transformed it
u(kₜ,hₜ) = log(wage(kₜ,hₜ)*hₜ) - ϕ * (hₜ^(1+γ))/(1+γ) # Stage game payoff - utility from wages times hours, disutility from hours worked

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
        u(kₜ, hₜ) + β*Eₜ(Ṽₜ₊₁, hₜ, kₜ)
    end
end

# Maximizes the stage game of the DP problem
function maximize_stage(Ṽₜ₊₁, kₜ, h_min, h_max)
    f = maximand(Ṽₜ₊₁, kₜ)
    maximize(f, h_min, h_max)
end

for t in T:-1:1
    Ṽₜ₊₁ = make_Ṽ(V, k_grid, t)
    (stage_k_min, stage_k_max) = get_state_boundary(t, h_min, h_max, ϵ_min, ϵ_max, k₁)
    i_max = findfirst(kⁱ -> kⁱ >= stage_k_max, k_grid)
    i_min = findlast(kⁱ -> kⁱ <= stage_k_min, k_grid)
    println("In stage $t we have (i_min,i_max)=$((i_min, i_max)) corresponding to $((k_grid[i_min], k_grid[i_max])), with actual boundaries $((stage_k_min, stage_k_max))")
    for i in i_min:i_max
        #if t==T && (i == i_min || i == i_max)
        #    continue
        #end
        kⁱ = k_grid[i]
        res = maximize_stage(Ṽₜ₊₁, kⁱ, h_min, h_max)
        policy[t, i] = maximizer(res)
        V[t, i] = maximum(res)

        @assert(maximum(res) != -Inf, "Grid point $i corresponding to $(kⁱ) in period $t has -Inf payoff")
    end
end


# Simulating result
shocks = sample(ϵ_grid, T)
hc = zeros(T+1)
hc[1] = k₁
hours = zeros(T)
wages = zeros(T)
incomes = zeros(T)

for t in 1:T
    cur_policy = interpolate((k_grid,), policy[t, :], Gridded(Linear()))
    hours[t] = cur_policy(hc[t])
    wages[t] = wage(hours[t], hc[t])
    hc[t+1] = transition(hc[t], hours[t], shocks[t])
    incomes[t] = hours[t]*wages[t]
end

# Plotting
plot(wages, label="Wages", legend=:bottomright, xlabel="Period", ylabel="Amount", title="Wages and Hours Worked Over Agent's Lifetime")
plot!(wages, label = "Wages")
plot!(hours[1:end-1], label="Savings")


