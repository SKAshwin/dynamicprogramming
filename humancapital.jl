# Build a model of (stochastic) human capital accumulation
# All wages consumed every period (no saving dynamic)
# Working more hours brings disutility, but increases wages in current period
# and adds to human capital levels
# More human capital increases wages
# Later add option to work 0 hours

using Interpolations, Optim, Distributions, Plots, StatsBase
using Optim: maximizer

@show Threads.nthreads()

# Represents a Human Capital Dynamic Programming problem
# Specifically, the exact parameters in the problem
# Also stores information about grids used
struct HumanCapitalDP
    T::Int
    k₁
    β
    A
    ρ
    θ
    γ
    λ
    ϕ
    σ
    shock_dist
    ϵ_min
    ϵ_max
    h_min
    h_max
    grid_k_min
    grid_k_max
    k_min
    k_max
    n_k_grid
    n_ϵ_grid::Int
    k_grid
    ϵ_grid
end

struct HumanCapitalDPSolution
    V
    policy
    policy_func
    V_func
    wage_func
    T
    k₁
    ϵ_grid
end

# Creates a HumanCapitalDP object with the specified parameter values
# In the process, it also creates the grids
# Edit this function if you want to change the grid size
# The grid in k is adaptive: it is more sparse the less likely a particular human capital value is likely to be reached
# Determined using a monte carlo simulation assuming uniformly played hours
function HumanCapitalDP(T, k₁, β, A, ρ, θ, γ, λ, ϕ, σ)
    n_k_adaptive_grid = 500 # The size the of the *adaptive* portion of the grid in k
    n_ϵ_grid = 20
    h_min = 0 
    h_max = 60
    shock_dist = LogNormal(0, σ)

    ϵ_grid = [quantile(shock_dist, i/(1+n_ϵ_grid)) for i in 1:n_ϵ_grid]
    ϵ_max = ϵ_grid[n_ϵ_grid]
    ϵ_min = ϵ_grid[1]

    (grid_k_min, grid_k_max) = get_state_boundary(T+1, h_min, h_max, ϵ_min, ϵ_max, k₁) # Min/Max HC level at the start of period T+1
    (k_min, k_max) = get_state_boundary(T, h_min, h_max, ϵ_min, ϵ_max, k₁) # Min/Max HC level at the start of period T

    # println("We have human capital space from $k_min to $k_max")

    # Generate the adaptive grid in k-space via a series of monte carlo simulations
    # The adaptive grid is supplemented by a top grid and a bottom grid
    # Covering the space from the lowest possible k_min to the bottom of the generated adaptive grid
    # and the space from the highest possible k_max to the top of the generated adaptive grid
    # These points are just equally spaced
    mc_k_sampled = vec(mc_hc(T, ϵ_grid, k₁, h_min, h_max, 60, 10000)[2:end, :])
    adaptive_grid = quantile(mc_k_sampled, range(0, stop=1, length=n_k_adaptive_grid))[2:end-1]
    bottom_grid = range(grid_k_min, stop=minimum(adaptive_grid), length=50)
    top_grid = range(maximum(adaptive_grid), stop=grid_k_max, length=50)
    k_grid = sort(vcat(bottom_grid, adaptive_grid, top_grid, k_max, k_min))
    k_grid = unique(k_grid)
    n_k_grid = length(k_grid)

    HumanCapitalDP(T, k₁, β, A, ρ, θ, γ, λ, ϕ, σ, shock_dist, ϵ_min, ϵ_max, h_min, h_max, grid_k_min, grid_k_max, 
                            k_min, k_max, n_k_grid, n_ϵ_grid, k_grid, ϵ_grid)
end

function solve(hcdp::HumanCapitalDP)
    V = zeros(hcdp.T+1, hcdp.n_k_grid) # V[t,i] is Vₜ(kⁱ) - the value function if you start period t with human capital of kⁱ
    policy = zeros(hcdp.T, hcdp.n_k_grid) # policy[t,i] is the optimal hours worked if at period t you have human capital kⁱ
    for t in hcdp.T:-1:1
        Ṽₜ₊₁ = make_Ṽ(V, hcdp.k_grid, t+1)
        (stage_k_min, stage_k_max) = get_state_boundary(t, hcdp.h_min, hcdp.h_max, hcdp.ϵ_min, hcdp.ϵ_max, hcdp.k₁)
        i_max = findfirst(kⁱ -> kⁱ >= stage_k_max, hcdp.k_grid)
        i_min = findlast(kⁱ -> kⁱ <= stage_k_min, hcdp.k_grid)
        #println("In stage $t we have (i_min,i_max)=$((i_min, i_max)) corresponding to $((k_grid[i_min], k_grid[i_max])), with actual boundaries $((stage_k_min, stage_k_max))")
        Threads.@threads for i in i_min:i_max
            kⁱ = hcdp.k_grid[i]
            res = maximize_stage(hcdp, Ṽₜ₊₁, kⁱ)
            policy[t, i] = maximizer(res)
            V[t, i] = maximum(res)
    
            @assert(maximum(res) != -Inf, "Grid point $i corresponding to $(kⁱ) in period $t has -Inf payoff")
        end
    end
    policy_func = interpolate((1:hcdp.T,hcdp.k_grid), policy, (NoInterp(),Gridded(Linear())))
    V_func = interpolate((1:hcdp.T+1,hcdp.k_grid), V, (NoInterp(),Gridded(Linear())))
    wage_func = (k, h) -> wage(hcdp, k, h)
    return HumanCapitalDPSolution(V, policy, policy_func, V_func, wage_func, hcdp.T, hcdp.k₁, hcdp.ϵ_grid)
end

# Simulates the DP solution. If noise=true, the agent experiences the random shock to human capital every period
# If noise=false, no shocks occur (although, note the agent behaves as-if shocks can happen every period - these shocks
# merely always realize to 1, but the agent does not know this)
function simulate(solution::HumanCapitalDPSolution, noise::Bool)
    shocks = noise ? rand(solution.ϵ_grid, solution.T) : ones(solution.T)
    hc = zeros(solution.T+1)
    hc[1] = solution.k₁
    hours = zeros(solution.T)
    wages = zeros(solution.T)
    incomes = zeros(solution.T)

    for t in 1:solution.T
        hours[t] = solution.policy_func(t, hc[t])
        wages[t] = solution.wage_func(hc[t], hours[t])
        hc[t+1] = transition(hc[t], hours[t], shocks[t])
        incomes[t] = hours[t]*wages[t]
    end

    return (hours, wages, incomes, hc, shocks)
end

# Simulates N agents, returns tuple of matrices of hours, wages, incomes, human capital, and shocks
# See above for what noise does
function simulate(solution::HumanCapitalDPSolution, noise::Bool, N::Int)
    shocks = noise ? rand(solution.ϵ_grid, solution.T, N) : ones(solution.T, N)
    hc = zeros(solution.T+1,N)
    hc[1, :] .= solution.k₁
    hours = zeros(solution.T,N)
    wages = zeros(solution.T, N)
    incomes = zeros(solution.T, N)

    for t in 1:solution.T
        hours[t, :] = solution.policy_func.(t, hc[t, :])
        wages[t, :] = solution.wage_func.(hc[t, :], hours[t, :])
        hc[t+1, :] = transition.(hc[t, :], hours[t, :], shocks[t, :])
        incomes[t, :] = hours[t, :].*wages[t, :]
    end

    return (hours, wages, incomes, hc, shocks)
end

# Runs a monte carlo simulation of N realizations of the state after T periods, assuming a uniform distribution from hours worked
# Returns the (Tx1)xN matrix of human capital realizations, where the (t,n) entry is the amount of human capital in period t of the
# nth monte carlo simulation
function mc_hc(T, ϵ_grid, k₁, h_min, h_max, h_points, N)
    # Each column of the matrix is a different simulation
    shocks = rand(ϵ_grid, T, N)
    hours_worked = rand(range(h_min, stop=h_max, length=h_points), T, N)
    hc= zeros(T+1, N)
    hc[1,:].= k₁
    for t in 1:T
        hc[t+1, :] = transition.(hc[t,:], hours_worked[t,:], shocks[t,:])
    end
    return hc
end

# Get the maximum and minimum possible human capital in period t
function get_state_boundary(t, h_min, h_max, ϵ_min, ϵ_max, k₁)
    k_min = ϵ_min^(t-1)*k₁ + h_min * sum([(ϵ_max^(τ)) for τ in 1:t-1])
    k_max = ϵ_max^(t-1)*k₁ + h_max * sum([(ϵ_max^(τ)) for τ in 1:t-1])
    return (k_min, k_max)  
end

# Linearly interpolate the value function in period t
# returns Ṽₜ(kₜ)
function make_Ṽ(V, k_grid, t)
    interpolate((k_grid,), V[t, :], Gridded(Linear()))
end

# Eₜ(Ṽₜ₊₁(hₜ₊₁)|hₜ, kₜ)
# Expected interpolated value function in the next period given your current state and action
# As the state transition is stochastic
function Eₜ(hcdp::HumanCapitalDP, Ṽₜ₊₁, hₜ, kₜ)
    # Check the interpolated value function for every possible realization of the noise
    # and as such every possible realization of hₜ given the current state and action
    # average the results
    mean(map(ϵₜ -> Ṽₜ₊₁(transition(kₜ, hₜ, ϵₜ)), hcdp.ϵ_grid))
end

# The maximand in every period - the stage utility plus the expected value function of the next period
# given our actions this period and current state
function maximand(hcdp::HumanCapitalDP, Ṽₜ₊₁, kₜ)
    function(hₜ)
        u(hcdp, kₜ, hₜ) + hcdp.β*Eₜ(hcdp, Ṽₜ₊₁, hₜ, kₜ)
    end
end

# Maximizes the stage game of the DP problem
function maximize_stage(hcdp::HumanCapitalDP,Ṽₜ₊₁, kₜ)
    f = maximand(hcdp, Ṽₜ₊₁, kₜ)
    maximize(f, hcdp.h_min, hcdp.h_max)
end

wage(hcdp::HumanCapitalDP, kₜ,hₜ) = hcdp.A * kₜ^hcdp.ρ * hₜ^hcdp.θ
transition(kₜ, hₜ, ϵₜ) = (kₜ + hₜ)*ϵₜ
# Stage game payoff - utility from wages times hours, disutility from hours worked
# Utility and disutility are CRRA
u(hcdp::HumanCapitalDP,kₜ,hₜ) = crra(wage(hcdp, kₜ,hₜ)*hₜ, 1+hcdp.λ) - hcdp.ϕ * crra(hₜ, 1+hcdp.γ) 
crra(value, param) = (value^param)/param

# Common test values
# hcdp = HumanCapitalDP(20, 100, 0.9, 0.01, 0.6, 1.1, 0.33, -0.67, 0.17, 0.1)