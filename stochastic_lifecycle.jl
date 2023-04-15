using Interpolations, Optim, Distributions
using Optim: maximizer

# Change the deterministic code to allow for y to be randomly drawn 

# Make sure you have multi-threading enabled.
@show Threads.nthreads()

# Parameters
global const T = 50                   # Number of periods
global const r = 0.05                 # Interest rate
global const n_grid = 1000             # Number of grid points for the state (savings) space
global const β = 1/(1+r)           # Discount factor
global const min_consumption = 0.35    # Minimum consumption
# Minimum consumption must be set to more than the distance between adjacent grid points in savings space
# An assertion further below checks this - minimum consumption affects the max savings that can be in playing
# And so affects the total range represented by the grid, and so affects the grid space

global const income_μ = 1
global const income_σ = 0.1
global const income_n_points = 30
global const min_income = 0.5
global const max_income = 1.5
global const y_interval_points = range(min_income, stop=max_income, length=income_n_points+1)
global const y_intervals = [(y_interval_points[i], y_interval_points[i + 1]) for i in 1:length(y_interval_points) - 1]
global const y_grid = map(pair -> mean(Truncated(Normal(income_μ, income_σ), pair[1], pair[2])), y_intervals)
global const y_dist = Truncated(Normal(income_μ, income_σ), min_income, max_income)


# What are the odds of drawing yₜ, conditional on yₜ₋₁ being observed? 
# For the current specification, yₜ₋₁ gives no information, but we can try a random walk
# next
# This function uses global constants, which don't need to be passed in, which has no performance issues
function conditional_pmf(yₜ, yₜ₋₁)
    # Now, yₜ = 0.5*μ + 0.5yₜ₋₁ + ϵ
    # Where ϵ ~ N(0, σ²)
    # But with the distribution truncated
    # conditional_y_dist = 0.5*income_μ + Truncated(Normal(0.5*yₜ₋₁, income_σ), min_income, max_income)
    # Find out what interval on the income grid yₜ lies in 
    if yₜ == min_income
        return cdf(y_dist, min_income)
    end
    end_index = searchsortedfirst(y_interval_points, yₜ)
    start_index = end_index -1
    cdf(y_dist, y_interval_points[end_index]) - cdf(y_dist, y_interval_points[start_index])
end

# Makes the interpolated expected value function, given the gridded expected value function
# We have EV[t, i, j] for Eₜ₋₁(Vₜ(aⁱ, yₜ)|yʲ), but we need Eₜ₋₁(Ṽₜ(aₜ, yₜ)|yʲ) for continuous aₜ 
# We can do this by, for each discrete yʲ, we (linearly) interpolate the expected value function
# Since expected value is just a linear combination of the value function at several values of yₜ, 
# linearly interpolating this way is fine
# Interpolations only over the grid in savings, not the grid in income_n_points
# To be more specific, interpolates over the grid in income_n_points using "Constant"
# So returns a Ṽₜ(aₜ, yₜ), which is continuous in aₜ but takes discrete values of yₜ
function make_EṼ(EV, grid, y_grid, t)
    interpolate((grid,y_grid),EV[t, :, :], (Gridded(Linear()), Gridded(Constant())))
end

# This function returns a function of the choice variable (c)
# The payoff function + value function, being maximized in each step of the DP problem
# u(c) + βEṼ(aₜ₊₁, yₜ)
# where aₜ₊₁ = transition(a, c, y)
function maximand(EṼ, aₜ, yₜ)
    function(cₜ)
        if cₜ <=0 zero(cₜ)
            return -Inf
        end
        aₜ₊₁ = transition(aₜ, cₜ, yₜ)
        return u(cₜ) + β*EṼ(aₜ₊₁, yₜ)
    end
end

# Gives the maximum future discounted savings in the *worst case*
# income realizations
function max_future_discounted_savings(t, T, y_min, c_min, r)
    if t==T
        return 0
    else
        sum((y_min-c_min) / (1 + r)^k for k in 1:T - t)
    end
end

# Gives the maximum savings you could currently have in the
# *best case* income realizations, consuming the minimum
function max_current_savings(t, y_max, c_min, r)
    t<=1 ? 0 : sum((y_max-c_min) * (1+r)^k for k in 1:t-1)
end

# Maximize the stage game, given the interpolated value function
# for the next period, the current state, current income (shocks)
# and the boundaries of the action space (c_min and c_max)
function maximize_stage(Ṽ, aₜ, yₜ, c_min, c_max)
    # We now need to calculate what consumption would maximize the DP maximand, given that aₜ = aⁱ
    # ie the choice-specific payoff, conditional on optimal payoff function in the next period
    f = maximand(Ṽ, aₜ, yₜ)
    # We need to know what the maximum/min consumption allowed is
    maximize(f, c_min, c_max)
end

# Given a minimum required consumption, the current period and assets and current income realization yₜ
# Returns the minimum consumption and maximum consumption possible
# max consumption ensures even if we get the worst possible income outcome in every
# subsequent period, we will still avoid ending the game in negative savings
function consumption_boundary(aₜ, yₜ ,y_min, t, T, c_min, r)
    max_consumption = aₜ + yₜ + max_future_discounted_savings(t,T, y_min, c_min, r)
    max_consumption = c_min > max_consumption ? c_min : max_consumption
    return (c_min, max_consumption)
end

# Given the current period, total game length, max and min income, minimum consumption and
# return on assets, calculates, for this period, the maximum savings
# and minimum savings allowed
function state_boundary(t, T, y_min, y_max, c_min, r)
    a_max = max_current_savings(t, y_max, c_min, r)
    a_min = min_consumption-y_min-max_future_discounted_savings(t, T, y_min, c_min, r)
    return (a_min, a_max)
end

# Utility function
u(c) = log(c)
# Transition function for savings a given spending c
transition(a, c, y) = (1 + r) * (a + y - c)


# Value iteration
EV = zeros(T + 1, n_grid, income_n_points) # creates a (Tx1)x(n_grid)x(income_n_points) 0-tensor
# EV[t, i, j] stores the expected value function for having state (savings) i in period t with income realization j in period t-1 
# ie Eₜ₋₁(Vₜ(aⁱ, yₜ)|yʲ)
# Note the j is for the income realization in the *previous* period if yₜ is a markov process (right now, it is IID)
policy = zeros(T, n_grid, income_n_points)
# policy[t, i, j] stores the optimal consumption in period t, if you have aₜ=aⁱ assets at the start of period t and observe income yₜ=yʲ
# j means something different than above! yₜ=yʲ, not yₜ₋₁ = yʲ.
max_aₜ = max_current_savings(T, max_income, min_consumption, r)
min_aₜ = min_consumption-min_income-max_future_discounted_savings(1, T, min_income, min_consumption, r)
println(max_aₜ)
println(min_aₜ)
grid = collect(range(min_aₜ-1, stop=max_aₜ+1, length=n_grid))   # Grid in assets

# For testing
#EV[51, :, :] = map(((a,y),)-> -0.04*y*a^2+2*a, Iterators.product(grid, y_grid))
#maximands = [a -> EṼ(a, y) for y in y_grid]
#results = map(f -> maximize(f, min_aₜ, max_aₜ), maximands)

println("Minimum consumption is $min_consumption and distance between gridpoints is $((max_aₜ-min_aₜ)/n_grid)")
@assert((max_aₜ-min_aₜ)/n_grid <= min_consumption)

# Loop through each period backwards
for t in T:-1:1
    # Create the interpolated expected value function for the next period
    EṼ = make_EṼ(EV, grid, y_grid, t+1)

    # Calculate feasible portion of the grid in this period
    (a_min, a_max) = state_boundary(t, T, min_income, max_income, min_consumption, r)
    i_max = findfirst(aⁱ -> aⁱ > a_max, grid)
    i_min = findlast(aⁱ -> aⁱ < a_min, grid)
    EV[t, :, : ] = fill!(EV[t, :, :], -Inf)
    policy[t,:, :] = fill!(policy[t,:,:], min_consumption) 
    #println("i_min in $t is $i_min corresponding to $(grid[i_min])")
    #println("i_max in $t is $i_max corresponding to $(grid[i_max])")
    # Loop through each grid point for assets
    # Parallelize this section
    Threads.@threads for i in i_min:i_max
        aⁱ= grid[i]
        # Stores the grid for the value function in this period
        # Vᵢₜ[j] stores the value function, if you started period with assets aⁱ and drew income yʲ
        # ie, Vₜ(aⁱ, yʲ)
        Vᵢₜ = zeros(income_n_points)
        for j in 1:income_n_points
            yʲ = y_grid[j]
            # Get the max and minimum spending in the current period given the income realization
            # then maximize the stage game within these constraints
            (stage_c_min, stage_c_max) = consumption_boundary(aⁱ, yʲ, min_income, t, T, min_consumption, r)
            # if you're at i_min, which is barely outside of the boundary, relax the lower bound on consumption
            # to allow consumption that stays within the feasible asset space (min_consumption might be too high)
            stage_c_min = i == i_min ? 0 : stage_c_min
            
            # Given constraints, maximize stage game
            res = maximize_stage(EṼ, aⁱ, yʲ, stage_c_min, stage_c_max)
            policy[t, i, j] = maximizer(res)
            Vᵢₜ[j] = maximum(res)
            @assert(maximum(res) != -Inf, "Negative infinity Vₜ(aⁱ, yʲ) in state grid point (i,j) = ($i, $j), ie (aⁱ,yʲ)=$((aⁱ,yʲ)) in stage $t")
        end
        # We now want Eₜ₋₁(Vₜ(aⁱ, yₜ)|yʲ)
        # Which is ∑ Vₜ(aⁱ, yᵏ)Q(yʲ, yᵏ)
        # Where Q(yʲ, yᵏ) is the probability if drawing yᵏ in period t if you drew yʲ in period t-1
        # and Vₜ(aⁱ, yᵏ) = Vᵢₜ[k]
        for j in 1:income_n_points
            EV[t, i, j] = sum(map(k -> Vᵢₜ[k] * conditional_pmf(y_grid[k], y_grid[j]), 1:income_n_points))
        end
    end
    
    #println("i_min in $t is $i_min corresponding to $(grid[i_min]), with value function $(V[t, i_min]) and policy $(policy[t, i_min])")
    #println("i_max in $t is $i_max corresponding to $(grid[i_max]), with value function $(V[t, i_max]) and policy $(policy[t, i_max])")
end
