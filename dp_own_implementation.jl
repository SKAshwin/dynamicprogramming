using Interpolations, Optim
using Optim: maximizer

# Make sure you have multi-threading enabled.
@show Threads.nthreads()

# Parameters
T = 50                   # Number of periods
r = 0.05                 # Interest rate
n_grid = 3000             # Number of grid points for the state (savings) space
income = 0.5              # Deterministic income per period
β = 0.96                # Discount factor
min_consumption = 0.05     # Minimum consumption

# For testing
#V[T+1, :] = map(x->-0.04(x+25)^2 + 2(x+25), grid)

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

# Utility function
u(c) = log(c)

# Transition function for savings a given spending c
transition(a, c, y) = (1 + r) * (a + y - c)
# Given how many assets you have next period and how much income this period
# what was your consumption
invtransition(aₜ₊₁, aₜ, yₜ) = aₜ - yₜ - aₜ₊₁/(1+r)  
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

# Loop through each period backwards
for t in T:-1:1
    # Create the interpolated function for the next period
    Ṽ = make_Ṽ(V, grid, t+1)
    i_min = 0
    i_max = n_grid + 1
    # Loop through each grid point for assets
    for i in 1:n_grid
        aⁱ= grid[i]
        # Some grid points can never feasibly be reached, so we can safely ignore them
        # if the grid point is more than you could ever save
        # by this point, don't bother computing the value functions
        # Add some tolerance
        if aⁱ > max_current_savings(t, income, min_consumption, r)
            policy[t, i] = min_consumption
            V[t, i] = -Inf
            # if this is the first time this clause is triggered, assign i_max
            if i_max == n_grid + 1 
                i_max =  i
            end 
            continue
        elseif aⁱ < min_consumption-income-max_future_discounted_savings(t, T, income, min_consumption, r)
            # similarly, if this grid point corresponds to having so little savings
            # you can't consume the minimum level going forward, ignore it 
            policy[t, i] = min_consumption
            V[t, i] = -Inf 
            i_min = i
            continue
        end

        # We now need to calculate what consumption would maximize the DP maximand, given that aₜ = aⁱ
        # ie the choice-specific payoff, conditional on optimal payoff function in the next period
        #f = maximand(Ṽ, aⁱ, income)
        # We need to know what the maximum/min consumption allowed is
        #max_consumption = aⁱ + income + max_future_discounted_savings(t,T, income, min_consumption, r)
        #max_consumption = min_consumption > max_consumption ? min_consumption : max_consumption
        #res = maximize(f, min_consumption, max_consumption)
        (stage_c_min, stage_c_max) = consumption_boundary(aⁱ, income, t, T, min_consumption, r)
        res = maximize_stage(Ṽ, aⁱ, income, stage_c_min, stage_c_max)
        policy[t, i] = maximizer(res)
        V[t, i] = maximum(res)
    end
    # For the boundaries of the feasible region, the grid points just outside it are used
    # for Interpolations in the next iteration
    # if they are -Inf, you will progressively lose gridpoints at the edge of every iteration
    if i_min > 0
        println("i_min")
        println(t)
        println(i_min)
        f = maximand(Ṽ, grid[i_min], income)
        max_consumption = grid[i_min] + income + max_future_discounted_savings(t,T, income, min_consumption, r)
        max_consumption = min_consumption > max_consumption ? min_consumption : max_consumption
        res = maximize(f, 0, max_consumption)
        #res = maximize_stage(Ṽ, grid[i_min], income, t, T, 0, r)
        policy[t, i_min] = maximizer(res)
        V[t, i_min] = maximum(res)
        println(maximizer(res))
        println(maximum(res))
    end
    if i_max < n_grid +1
        println("i_max")
        println(t)
        println(i_max)
        V[t, i_max] = V[t, i_max - 1]
        policy[t,i_max] = policy[t, i_max - 1]
    end
end

