# Build a model of (stochastic) human capital accumulation
# All wages consumed every period (no saving dynamic)
# Working more hours brings disutility, but increases wages in current period
# and adds to human capital levels
# More human capital increases wages
# Distate to work changes randomly every period
# Change distate to work to ϕᵢ + ωᵢₜ for each agent; so agents have different systematic tastes to work
# Actually we already have that - just make a mixture
# That get shocked every period
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
    μ_ϕ
    σ_ϕ
    σ_ϵ
    ϵ_min
    ϵ_max
    h_min
    h_max
    grid_k_min
    grid_k_max
    n_k_grid::Int
    n_ϵ_grid::Int
    n_ϕ_grid::Int
    k_grid
    ϵ_grid
    ϕ_grid
end

struct HumanCapitalDPSolution
    EV
    policy
    policy_func
    EV_func
    wage_func
    T
    k₁
    μ_ϕ
    ϵ_grid
    ϕ_grid
end

# Creates a HumanCapitalDP object with the specified parameter values
# In the process, it also creates the grids
# Edit this function if you want to change the grid size
# The grid in k is adaptive: it is more sparse the less likely a particular human capital value is likely to be reached
# Determined using a monte carlo simulation assuming uniformly played hours
function HumanCapitalDP(T, k₁, β, A, ρ, θ, γ, λ, μ_ϕ, σ_ϕ, σ_ϵ)
    n_k_adaptive_grid = 100 # The size the of the *adaptive* portion of the grid in k
    n_ϵ_grid = 20
    n_ϕ_grid = 10
    h_min = 0 
    h_max = 120
    shock_dist = LogNormal(0, σ_ϵ)
    ϕ_dist = LogNormal(log(μ_ϕ), σ_ϕ) # μ_ϕ is the mean of the realized ϕ, so we pass its log here

    ϵ_grid = [quantile(shock_dist, i/(1+n_ϵ_grid)) for i in 1:n_ϵ_grid]
    ϵ_max = ϵ_grid[n_ϵ_grid]
    ϵ_min = ϵ_grid[1]

    ϕ_grid = [quantile(ϕ_dist, i/(1+n_ϕ_grid)) for i in 1:n_ϕ_grid]

    (grid_k_min, grid_k_max) = get_state_boundary(T+1, h_min, h_max, ϵ_min, ϵ_max, k₁) # Min/Max HC level at the start of period T+1
    boundary_points = collect(Iterators.flatten(get_state_boundary.(1:T, h_min, h_max, ϵ_min, ϵ_max, k₁))) 
    # Every edge of the boundary
    # in each period
    # should be in the final grid

    # println("We have human capital space from $k_min to $k_max")

    # Generate the adaptive grid in k-space via a series of monte carlo simulations
    # The adaptive grid is supplemented by a top grid and a bottom grid
    # Covering the space from the lowest possible k_min to the bottom of the generated adaptive grid
    # and the space from the highest possible k_max to the top of the generated adaptive grid
    # These points are just equally spaced
    mc_k_sampled = vec(mc_hc(T, ϵ_grid, k₁, h_min, h_max, 60, 10000)[2:end, :])
    adaptive_grid = quantile(mc_k_sampled, range(0, stop=1, length=n_k_adaptive_grid))[2:end-1]
    bottom_grid = range(grid_k_min, stop=minimum(adaptive_grid), length=25)
    top_grid = range(maximum(adaptive_grid), stop=grid_k_max, length=25)
    k_grid = sort(vcat(bottom_grid, adaptive_grid, top_grid, boundary_points))
    k_grid = unique(k_grid)
    n_k_grid = length(k_grid)

    HumanCapitalDP(T, k₁, β, A, ρ, θ, γ, λ, μ_ϕ, σ_ϕ, σ_ϵ, ϵ_min, ϵ_max, h_min, h_max, grid_k_min, grid_k_max, 
                            n_k_grid, n_ϵ_grid, n_ϕ_grid, k_grid, ϵ_grid, ϕ_grid)
end

function solve(hcdp::HumanCapitalDP)
    EV = fill(-Inf, hcdp.T+1, hcdp.n_k_grid)
    # V[t,i] is E(Ṽₜ(kⁱ, ϕₜ)) - the expected value function if you start period t with human capital of kⁱ
    # *Before* the realization of ϕₜ in that period
    # Fill everything with -Inf, so out-of-scope points are labelled correctly
    EV[hcdp.T+1, :] = zeros(hcdp.n_k_grid) # Period T+1 is full of 0s, though
    policy = zeros(hcdp.T, hcdp.n_k_grid, hcdp.n_ϕ_grid) # policy[t,i,j] is the optimal hours worked if at period t you have human capital kⁱ
    # and you have ϕₜ = ϕʲ
    for t in hcdp.T:-1:1
        EṼₜ₊₁ = make_EṼ(EV, hcdp.k_grid, t+1)
        (stage_k_min, stage_k_max) = get_state_boundary(t, hcdp.h_min, hcdp.h_max, hcdp.ϵ_min, hcdp.ϵ_max, hcdp.k₁)
        i_max = findfirst(kⁱ -> kⁱ >= stage_k_max, hcdp.k_grid)
        i_min = findlast(kⁱ -> kⁱ <= stage_k_min, hcdp.k_grid)
        #println("In stage $t we have (i_min,i_max)=$((i_min, i_max)) corresponding to $((hcdp.k_grid[i_min], hcdp.k_grid[i_max])), with actual boundaries $((stage_k_min, stage_k_max))")
        Threads.@threads for i in i_min:i_max
            kⁱ = hcdp.k_grid[i]
            @assert(!(kⁱ < stage_k_min || kⁱ > stage_k_max), "$(kⁱ) is out of the stage bounds $((stage_k_min, stage_k_max))")
            # Stores the grid for the value function in this period
            # Vᵢₜ[j] stores the value function, if you started period with human capital kⁱ and drew distate ϕₜ = ϕʲ
            # ie, Vₜ(kⁱ, ϕʲ)
            Vᵢₜ = zeros(hcdp.n_ϕ_grid)
            for j in 1:hcdp.n_ϕ_grid
                ϕʲ = hcdp.ϕ_grid[j]
                res = maximize_stage(hcdp, EṼₜ₊₁, kⁱ, ϕʲ)
                policy[t, i, j] = maximizer(res)
                Vᵢₜ[j] = maximum(res)
                @assert(maximum(res) != -Inf, "Grid point ($i, $j) corresponding to (kⁱ,ϕʲ) = ($(kⁱ), $(ϕʲ)) in period $t has -Inf payoff")
            end
            
            # Each realization of ϕʲ is equally likely by construction
            # We want E(Vₜ(kⁱ, ϕₜ)) which is
            # ∑Vₜ(kⁱ, ϕʲ)Pr(ϕₜ=ϕʲ)
            # which is just the arithmetic mean of the Vₜ(kⁱ, ϕʲ)
            EV[t, i] = mean(Vᵢₜ)
    
            @assert(EV[t, i] != -Inf, "Grid point $i corresponding to $(kⁱ) in period $t has -Inf payoff")
        end
    end
    # Interpolate over the continuous kₜ, but not the ϕₜ which are drawn from a discretized distribution
    policy_func = interpolate((1:hcdp.T,hcdp.k_grid, hcdp.ϕ_grid), policy, (NoInterp(),Gridded(Linear()),Gridded(Constant())))
    EV_func = interpolate((1:hcdp.T+1,hcdp.k_grid), EV, (NoInterp(),Gridded(Linear())))
    wage_func = (k, h) -> wage(hcdp, k, h)
    return HumanCapitalDPSolution(EV, policy, policy_func, EV_func, wage_func, hcdp.T, hcdp.k₁, hcdp.μ_ϕ, hcdp.ϵ_grid, hcdp.ϕ_grid)
end

# Simulates the DP solution. If noise=true, the agent experiences the random shock to human capital every period
# If ϵ_noise=false, no HC shocks occur (although, note the agent behaves as-if shocks can happen every period - these shocks
# merely always realize to 1, but the agent does not know this)
# Similarly, if ϕ_noise = false, no work distaste shocks occur - it is always realized at the mean distaste
function simulate(solution::HumanCapitalDPSolution, ϵ_noise::Bool, ϕ_noise::Bool)
    ϵ_realizations = ϵ_noise ? rand(solution.ϵ_grid, solution.T) : ones(solution.T)
    ϕ_realizations = ϕ_noise ? rand(solution.ϕ_grid, solution.T) : ones(solution.T) .* solution.μ_ϕ
    hc = zeros(solution.T+1)
    hc[1] = solution.k₁
    hours = zeros(solution.T)
    wages = zeros(solution.T)
    incomes = zeros(solution.T)

    for t in 1:solution.T
        hours[t] = solution.policy_func(t, hc[t], ϕ_realizations[t])
        wages[t] = solution.wage_func(hc[t], hours[t])
        hc[t+1] = transition(hc[t], hours[t], ϵ_realizations[t])
        incomes[t] = hours[t]*wages[t]
    end

    return (hours, wages, incomes, hc, ϵ_realizations, ϕ_realizations)
end

# Simulates N agents, returns tuple of matrices of hours, wages, incomes, human capital, and shocks
# See above for what the noise arguments do
function simulate(solution::HumanCapitalDPSolution, ϵ_noise::Bool, ϕ_noise::Bool, N::Int)
    ϵ_realizations = ϵ_noise ? rand(solution.ϵ_grid, solution.T, N) : ones(solution.T, N)
    ϕ_realizations = ϕ_noise ? rand(solution.ϕ_grid, solution.T, N) : ones(solution.T, N) .* solution.μ_ϕ
    hc = zeros(solution.T+1,N)
    hc[1, :] .= solution.k₁
    hours = zeros(solution.T,N)
    wages = zeros(solution.T, N)
    incomes = zeros(solution.T, N)

    for t in 1:solution.T
        hours[t, :] = solution.policy_func.(t, hc[t, :], ϕ_realizations[t, :])
        wages[t, :] = solution.wage_func.(hc[t, :], hours[t, :])
        hc[t+1, :] = transition.(hc[t, :], hours[t, :], ϵ_realizations[t, :])
        incomes[t, :] = hours[t, :].*wages[t, :]
    end

    return (hours, wages, incomes, hc, ϵ_realizations, ϕ_realizations)
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
    k_min = ϵ_min^(t-1)*k₁ + h_min * sum([(ϵ_min^(τ)) for τ in 1:t-1])
    k_max = ϵ_max^(t-1)*k₁ + h_max * sum([(ϵ_max^(τ)) for τ in 1:t-1])
    return (k_min, k_max)  
end

# Linearly interpolate the expected value function in period t
# returns E(Ṽₜ₊₁(kₜ₊₁, ϕₜ₊₁)|kₜ₊₁) ie where kₜ₊₁ is known and a continuous value
# EV[t, i] should have, for discrete kⁱ, E(Ṽₜ₊₁(kₜ₊₁, ϕₜ₊₁)|kₜ₊₁=kⁱ)
function make_EṼ(EV, k_grid, t)
    interpolate((k_grid, ), EV[t, :], Gridded(Linear()))
end

# Eₜ(Ṽₜ₊₁(kₜ₊₁, ϕₜ₊₁)|hₜ, kₜ)
# Expected interpolated value function in the next period given your current state and action
# As the state transition is stochastic
function Eₜ(hcdp::HumanCapitalDP, EṼₜ₊₁, hₜ, kₜ)
    # Check the interpolated value function for every possible realization of the noise
    # and as such every possible realization of hₜ given the current state and action
    # average the results
    mean(map(ϵₜ -> EṼₜ₊₁(transition(kₜ, hₜ, ϵₜ)), hcdp.ϵ_grid))
end

# The maximand in every period - the stage utility plus the expected value function of the next period
# given our actions this period and current state, and the realization of the stochastic ϕₜ
function maximand(hcdp::HumanCapitalDP, EṼₜ₊₁, kₜ, ϕₜ)
    function(hₜ)
        u(hcdp, kₜ, hₜ, ϕₜ) + hcdp.β*Eₜ(hcdp, EṼₜ₊₁, hₜ, kₜ)
    end
end

# Maximizes the stage game of the DP problem
# Given the current state and the realization of the stochastic ϕₜ
function maximize_stage(hcdp::HumanCapitalDP, EṼₜ₊₁, kₜ, ϕₜ)
    f = maximand(hcdp, EṼₜ₊₁, kₜ, ϕₜ)
    maximize(f, hcdp.h_min, hcdp.h_max)
end

wage(hcdp::HumanCapitalDP, kₜ,hₜ) = hcdp.A * kₜ^hcdp.ρ * hₜ^hcdp.θ
transition(kₜ, hₜ, ϵₜ) = (kₜ + hₜ)*ϵₜ
# Stage game payoff - utility from wages times hours, disutility from hours worked
# Utility and disutility are CRRA
# Need to give the realization of the stochastic ϕₜ
u(hcdp::HumanCapitalDP,kₜ,hₜ,ϕₜ) = crra(wage(hcdp, kₜ,hₜ)*hₜ, 1+hcdp.λ) - ϕₜ * crra(hₜ, 1+hcdp.γ) 
crra(value, param) = (value^param)/param

# Common test values
# hcdp = HumanCapitalDP(20, 100, 0.9, 0.01, 0.6, 1.1, 0.33, -0.67, 0.17, 0.1)