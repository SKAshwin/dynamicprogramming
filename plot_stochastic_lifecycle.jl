using Plots, StatsBase

# Simulation
savings = zeros(T + 1)
incomes = zeros(T) # Random savings draws
savings[1] = 0  # Initial savings
incomes[1] = y_grid[floor(Int, income_n_points/2)] # initial income
consumptions = zeros(T)

for t in 1:T
    # Interpolate the policy function for the current state (savings) and income
    if t != 1
        probabilities = map(j->conditional_pmf(y_grid[j], incomes[t-1]), 1:income_n_points)
        conditional_dist = Weights(probabilities)
        incomes[t] = sample(y_grid, conditional_dist)
    end
    current_policy = interpolate((grid,y_grid), policy[t, :, :], (Gridded(Linear()), Gridded(Constant())))
    consumptions[t] = current_policy(savings[t], incomes[t])
    savings[t + 1] = transition(savings[t], consumptions[t], incomes[t])
    if t < T && savings[t+1] < -min_income-max_future_discounted_savings(t+1, T, min_income, 0, r)
        # can happen due to savings being too close to minimum
        # specifically, past the grid point that defines the minimum
        # recommended consumption policy results in going beyond minimum savings
        # default to minimum consumption in this case
        println("In period $t you were risking negative final savings, with savings $(savings[t+1]) at the start of period $(t+1)")
        #error("Negative final assets")
    end
end

# Plotting
plot(consumptions, label="Consumption", legend=:bottomright, xlabel="Period", ylabel="Amount", title="Consumption and Savings Over Agent's Lifetime")
plot!(incomes, label = "Income")
plot!(savings[1:end-1], label="Savings")