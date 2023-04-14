using Interpolations, Plots, LinearAlgebra

# Make sure you have multi-threading enabled.
@show Threads.nthreads()

# Parameters
T = 50                   # Number of periods
r = 0.05                 # Interest rate
n_grid = 100             # Number of grid points for the state (savings) space
n_income = 5             # Number of grid points for the stochastic income
min_income = 0.5
income_values = range(min_income, stop=1.0, length=n_income)  # Income values
income_probs = fill(1/n_income, n_income)             # Income probabilities (uniform distribution)
beta = 0.96                # Discount factor
min_consumption = 0.1     # Minimum consumption

# Utility function
u(c) = log(c)

# Transition function for savings given action (spending)
transition(s, c, y) = (1 + r) * (s - c) + y

# Value iteration
V = zeros(T + 1, n_grid)
policy = zeros(T, n_grid, n_income)
grid = range(0.0, stop=1.0, length=n_grid)

for t in T:-1:1
    Threads.@threads for i in 1:length(grid)
        s = grid[i]
        for (j, y) in enumerate(income_values)
            max_c = 0
            max_value = -Inf

            if t < T
                # Set the lower and upper limits for spending
                lower_limit = min_consumption
                required_savings = sum((min_consumption-min_income) / (1 + r)^k for k in 0:T - t - 1)
                upper_limit = s + y - required_savings
                if upper_limit > 0
                    for c in range(lower_limit, stop=upper_limit, length=100)  # Iterate over spending
                        next_s = transition(s, c, y)
                        # Interpolate the value function for the next period's state (savings)
                        V_next = extrapolate(interpolate((grid,), V[t + 1, :], Gridded(Linear())), Flat())
                        expected_value = sum(income_probs[k] * V_next(next_s) for k in 1:n_income)
                        current_value = u(c) + beta * expected_value

                        if current_value > max_value
                            max_value = current_value
                            max_c = c
                        end
                    end
                end
            else
                # In the last period, the agent consumes all savings and income
                max_c = s + y
                if max_c < 0
                    max_value = -Inf
                else
                    max_value = u(max_c)
                end 
            end

            V[t, i] = max_value
            policy[t, i, j] = max_c
        end
    end
end

# Now, the value function `V` and the policy function `policy` have been calculated.