using Plots

# Simulation
savings = zeros(T + 1)
savings[1] = 0  # Initial savings
income = 0.5 # deterministic income realization
consumptions = zeros(T)

for t in 1:T
    # Interpolate the policy function for the current state (savings) and income
    current_policy = interpolate((grid,), policy[t, :], Gridded(Linear()))
    consumptions[t] = current_policy(savings[t])
    savings[t + 1] = transition(savings[t], consumptions[t], income)
    if t < T && savings[t+1] < -income-max_future_discounted_savings(t+1, T, income, 0, r)
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
plot!(savings[1:end-1], label="Savings")