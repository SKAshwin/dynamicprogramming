# Allow for many different agents with *different* Human Capital DP problems
# ie different parameters, randomly drawn
# Solve all of them 
# Observe moments on the results
# Allow for individual-specific distate in work to be drawn from a log-normal distribution
# To be precise, ϕᵢₜ = ϕᵢωᵢₜ, where ϕᵢ ~ logNormal(ln(μ_ϕ), σ_ϕ)
# and ωᵢₜ = logNormal(0, σ_ω) 
# WARNING: This μ_ϕ is different from the μ_ϕ in HumanCapitalDP. μ_ϕ here is the *population* mean
# of ϕ, whereas in HumanCapitalDP it is the individual mean of ϕ. 
# Also allow A, the overall productivity agents are endowed with, for a given level of human capital and hours worked
# to vary by person, be drawn from a log-normal distribution

# For testing, this is close to the real thing
# mix = HumanCapitalDPMixture(16, 50, 100, 0.9, 0.01, 0.3, 0.6, 1.1, 0.33, -0.67, 0.15, 0.1, 0.1, 0.01)

using Distributions
include("humancapital.jl")

# Human Capital DP Mixture consists of a set of N agents
# Who have the given parameters, and varying work distaste parameters drawn from a Log-Normal distribution
# The draws occur upon calling solve - the mixture itself just defines the parameters of the mix, not a realization
struct HumanCapitalDPMixture
    T::Int
    N::Int
    k₁
    β
    μ_A
    σ_A
    ρ
    θ
    γ
    λ
    μ_ϕ   # The population mean of the individual-fixed distaste for work, unlike in HumanCapitalDP!
    σ_ϕ   # The population variance in the individual-fixed distate for work.
    σ_ω   # Corresponds to σ_ϕ in HumanCapitalDP
    σ_ϵ
end

struct HumanCapitalDPMixtureSolution
    T::Int
    N::Int
    A_realizations
    ϕ_realizations
    solutions::Vector{HumanCapitalDPSolution}
end

function solve(mixture::HumanCapitalDPMixture)
    ϕ_dist = LogNormal(log(mixture.μ_ϕ), mixture.σ_ω) # TODO: Remember to fix this
    ϕ_realizations = rand(ϕ_dist, mixture.N)
    A_dist = LogNormal(log(mixture.μ_A), mixture.σ_A)
    A_realizations = rand(A_dist, mixture.N)
    sols = Vector{HumanCapitalDPSolution}(undef, mixture.N)
    Threads.@threads for i in 1:mixture.N
        hcdp = HumanCapitalDP(mixture.T, mixture.k₁, mixture.β, A_realizations[i], mixture.ρ, mixture.θ, mixture.γ, mixture.λ, ϕ_realizations[i], mixture.σ_ω, mixture.σ_ϵ)
        sols[i] = solve(hcdp)
    end
    HumanCapitalDPMixtureSolution(mixture.T, mixture.N, A_realizations, ϕ_realizations,sols)
end

# Simulates the mixture of agents in the solved HumanCapitalDPMixture
# The each have a (realized) stochastic mean ϕ, distaste for work.
# ϕ_noise controls whether you want this disaste for work to vary across time, within each agent
# ϵ_noise controls whether you want shocks to the total amount of human capital every period
function simulate(mix_solution::HumanCapitalDPMixtureSolution, ϵ_noise::Bool, ϕ_noise::Bool)
    solutions = mix_solution.solutions
    ϵ_realizations = ones(mix_solution.T, mix_solution.N)
    ϕ_realizations = ones(mix_solution.T, mix_solution.N)
    hc = zeros(mix_solution.T+1, mix_solution.N)
    hours = zeros(mix_solution.T, mix_solution.N)
    wages = zeros(mix_solution.T, mix_solution.N)
    incomes = zeros(mix_solution.T, mix_solution.N)
    Threads.@threads for i in 1:mix_solution.N
        hc[1, i] = solutions[i].k₁
        if ϵ_noise
            ϵ_realizations[:, i] = rand(solutions[i].ϵ_grid, mix_solution.T)
        end

        if ϕ_noise
            ϕ_realizations[:, i] = rand(solutions[i].ϕ_grid, mix_solution.T) 
        else
            ϕ_realizations[:, i] = ones(mix_solution.T) .* solutions[i].μ_ϕ
        end
        
        for t in 1:mix_solution.T
            hours[t, i] = solutions[i].policy_func(t, hc[t, i], ϕ_realizations[t, i])
            wages[t, i] = solutions[i].wage_func(hc[t, i], hours[t, i])
            hc[t+1, i] = transition(hc[t, i], hours[t, i], ϵ_realizations[t, i])
            incomes[t, i] = hours[t, i]*wages[t, i]
        end
    end

    return (hours, wages, incomes, hc, ϵ_realizations, ϕ_realizations)
end
