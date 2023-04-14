using Interpolations, Optim
using Optim: maximizer

# Make sure you have multi-threading enabled.
@show Threads.nthreads()

# Parameters
T = 50                   # Number of periods
r = 0.05                 # Interest rate
n_grid = 500             # Number of grid points for the state (savings) space
income = 0.5              # Deterministic income per period
β = 0.95                # Discount factor
min_consumption = 0.2     # Minimum consumption

# Minimum consumption must be set to more than the distance between adjacent grid points in savings space
# An assertion further below checks this - minimum consumption affects the max savings that can be in playing
# And so affects the total range represented by the grid, and so affects the grid space

# Makes the interpolated value function, given the gridded value function
function make_Ṽ(V, grid, t)
    interpolate((grid,),V[t, :], Gridded(Linear()))
end

# This function returns a function of the choice variable (c)
# The payoff function + value function, being maximized in each step of the DP problem
# u(c) + βṼ(aₜ₊₁)
# where aₜ₊₁ = transition(a, c, y)
function maximand(Ṽ, aₜ, yₜ)
    function(cₜ)
        if cₜ < zero(cₜ)
            return -Inf
        end
        aₜ₊₁ = transition(aₜ, cₜ, yₜ)
        return u(cₜ) + β*Ṽ(aₜ₊₁)
    end
end

function max_future_discounted_savings(t, T, y, c_min, r)
    if t==T
        return 0
    else
        sum((y-c_min) / (1 + r)^k for k in 1:T - t)
    end
end

function max_current_savings(t, y, c_min, r)
    t<=1 ? 0 : sum((y-c_min) * (1+r)^k for k in 1:t-1)
end

# Maximize the stage game, given the interpolated value function
# for the next period, the current state, current income (shocks)
# and the boundaries of the action space (c_min and c_max)
function maximize_stage(Ṽ, aₜ, y, c_min, c_max)
    # We now need to calculate what consumption would maximize the DP maximand, given that aₜ = aⁱ
    # ie the choice-specific payoff, conditional on optimal payoff function in the next period
    f = maximand(Ṽ, aₜ, y)
    # We need to know what the maximum/min consumption allowed is
    maximize(f, c_min, c_max)
end

# Given a minimum required consumption, the current period and assets
# Returns the minimum consumption and maximum consumption possible
function consumption_boundary(aₜ, y, t, T, c_min, r)
    max_consumption = aₜ + y + max_future_discounted_savings(t,T, y, c_min, r)
    max_consumption = c_min > max_consumption ? c_min : max_consumption
    return (c_min, max_consumption)
end

# Given the current period, total game length, income, minimum consumption and
# return on assets, calculates, for this period, the maximum savings
# and minimum savings allowed
function state_boundary(t, T, y, c_min, r)
    a_max = max_current_savings(t, y, c_min, r)
    a_min = min_consumption-income-max_future_discounted_savings(t, T, y, c_min, r)
    return (a_min, a_max)
end

# Utility function
u(c) = log(c)
# Transition function for savings a given spending c
transition(a, c, y) = (1 + r) * (a + y - c)


# Value iteration
V = zeros(T + 1, n_grid) # creates a (Tx1)x(n_grid) 0-matrix
# V[t, i] stores the value function for playing action i in period t - ie Vₜ(aⁱ)
policy = zeros(T, n_grid)
# policy[t, i] stores the optimal consumption in period t, if you have aₜ=aⁱ assets at the start of period t
max_aₜ = max_current_savings(T, income, min_consumption, r) + 0.1
min_aₜ = min_consumption-income-max_future_discounted_savings(1, T, income, min_consumption, r)
println(max_aₜ)
println(min_aₜ)
grid = collect(range(min_aₜ-1, stop=max_aₜ+1, length=n_grid))   # Grid in assets

println("Minimum consumption is $min_consumption and distance between gridpoints is $((max_aₜ-min_aₜ)/n_grid)")
@assert((max_aₜ-min_aₜ)/n_grid <= min_consumption)

# Loop through each period backwards
for t in T:-1:1
    # Create the interpolated function for the next period
    Ṽ = make_Ṽ(V, grid, t+1)

    # Calculate feasible portion of the grid in this period
    (a_min, a_max) = state_boundary(t, T, income, min_consumption, r)
    i_max = findfirst(aⁱ -> aⁱ > a_max, grid)
    i_min = findlast(aⁱ -> aⁱ < a_min, grid)
    V[t, :] = fill!(V[t, :], -Inf)
    policy[t,:] = fill!(policy[t,:], min_consumption) 
    # Loop through each grid point for assets
    # Parallelize this section
    Threads.@threads for i in i_min:i_max
        aⁱ= grid[i]
        # Get the max and minimum spending in the current period
        # then maximize the stage game within these constraints
        (stage_c_min, stage_c_max) = consumption_boundary(aⁱ, income, t, T, min_consumption, r)
        # if you're at i_min, which is barely outside of the boundary, relax the lower bound on consumption
        # to allow consumption that stays within the feasible asset space (min_consumption might be too high)
        stage_c_min = i == i_min ? 0 : stage_c_min
        
        # Given constraints, maximize stage game
        res = maximize_stage(Ṽ, aⁱ, income, stage_c_min, stage_c_max)
        policy[t, i] = maximizer(res)
        V[t, i] = maximum(res)
    end
    
    #println("i_min in $t is $i_min corresponding to $(grid[i_min]), with value function $(V[t, i_min]) and policy $(policy[t, i_min])")
    #println("i_max in $t is $i_max corresponding to $(grid[i_max]), with value function $(V[t, i_max]) and policy $(policy[t, i_max])")
end
