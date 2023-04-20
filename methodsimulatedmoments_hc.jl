using Optim, DataFrames, CSV, BlackBoxOptim
include("humancapital.jl")
include("humancapitalDPmixture.jl")

# Returns the median, mean and variance of hours worked and wages by period
# From monte carlo simulations
# In a DataFrame
function moments_by_period(sol::HumanCapitalDPSolution, N::Int)
    (hours, wages, _, _, _) = simulate(sol, true, true, N)
    mean_hours = vec(mapslices(mean, hours, dims=2))
    med_hours = vec(mapslices(median, hours, dims=2))
    mad_hours = vec(mapslices(mad_slice, hours, dims=2))
    
    mean_wages = vec(mapslices(mean, wages, dims=2))
    med_wages = vec(mapslices(median, wages, dims=2))
    mad_wages = vec(mapslices(mad_slice, wages, dims=2))

    return DataFrame(mean_hours = mean_hours, med_hours = med_hours, mad_hours = mad_hours,
                        mean_wage = mean_wages, med_wage = med_wages, mad_wage = mad_wages)
end

# Given a mixture solution, calculates moments by period
function moments_by_period(mixsol::HumanCapitalDPMixtureSolution)
    (hours, wages, _, _, _) = simulate(mixsol, true, true)
    mean_hours = vec(mapslices(mean, hours, dims=2))
    med_hours = vec(mapslices(median, hours, dims=2))
    mad_hours = vec(mapslices(mad_slice, hours, dims=2))
    
    mean_wages = vec(mapslices(mean, wages, dims=2))
    med_wages = vec(mapslices(median, wages, dims=2))
    mad_wages = vec(mapslices(mad_slice, wages, dims=2))

    return DataFrame(mean_hours = mean_hours, med_hours = med_hours, mad_hours = mad_hours,
                        mean_wage = mean_wages, med_wage = med_wages, mad_wage = mad_wages)
end


# Given an arbitrary vector/collection, calculates the median absolute deviation of that vector
# IE, for each entry, calculate its absolute deviation from the median of the collections
# And take the median of this
# This is a more robust measure of spread than standard deviation
function mad_slice(x)
    med = median(x)
    return median(abs.(x .- med))
end

# OLD MODEL iterations
# hcdp = HumanCapitalDP(16, 100, 0.9, 0.01, 0.6, 1.1, 0.33, -0.67, 0.17, 0.1)
# hcdp = HumanCapitalDP(16, 100, 0.9, 0.01, 0.6, 1.1, 0.33, -0.67, 0.15453838348388677, 0.1)
# Best candidate found: [0.919399, 0.735726, 0.913193] for [β, ρ, θ] Fitness: 84.809515475
# hcdp = HumanCapitalDP(16, 100, 0.919399, 0.01, 0.735726, 0.913193, 0.33, -0.67, 0.15453838348388677, 0.1)
# Best candidate found: [0.011465] Fitness: 0.000014239
# hcdp = HumanCapitalDP(16, 100, 0.919399, 0.011465, 0.735726, 0.913193, 0.33, -0.67, 0.15453838348388677, 0.1)
# hcdp = HumanCapitalDP(16, 100, 0.660972, 0.362691, 0.27416, 0.623034, 0.192238, -0.575506, 0.325281, 0.347913)
# function HumanCapitalDP(T, k₁, β, A, ρ, θ, γ, λ, ϕ, σ)

file_path = "population_moments.csv"
data_moments = CSV.read(file_path, DataFrame)

function moment_score(sol::HumanCapitalDPSolution, data_moments, N)
    model_moments = moments_by_period(sol, N)
    model_med_wages = model_moments[!, "med_wage"]
    data_med_wages = data_moments[!, "med_wage"]

    model_med_hours = model_moments[!, "med_hours"]
    data_med_hours = data_moments[!, "med_hours"]

    model_mad_hours = model_moments[!, "mad_hours"]
    data_mad_hours = data_moments[!, "mad_hours"]

    model_mad_wages = model_moments[!, "mad_wage"]
    data_mad_wages = data_moments[!, "mad_wage"]

    return sum((model_med_wages .- data_med_wages).^2) + sum((model_med_hours .- data_med_hours).^2) +
            sum((model_mad_wages .- data_mad_wages).^2) + sum((model_mad_hours .- data_mad_hours).^2)
end


#=
function all_tuner(hcdp::HumanCapitalDP, data_moments, N)
    function maximand(β, A, ρ, θ, γ, λ, ϕ, σ)
        new_hcdp = HumanCapitalDP(hcdp.T, hcdp.k₁, β, A, ρ, θ, γ, λ, ϕ, σ)
        sol = solve(new_hcdp)
        moment_score(sol, data_moments, N)
    end
end

function ϕ_moment_score(sol::HumanCapitalDPSolution, data_moments, N)
    start_hour_model = moments_by_period(sol, N)[1, "med_hours"]
    start_hour_data = data_moments[1, "med_hours"]
    return (start_hour_data-start_hour_model)^2
end

function ϕ_tuner(hcdp::HumanCapitalDP, data_moments, N)
    function maximand(ϕ)
        new_hcdp = HumanCapitalDP(hcdp.T, hcdp.k₁, hcdp.β, hcdp.A, hcdp.ρ, hcdp.θ, hcdp.γ, hcdp.λ, ϕ, hcdp.σ)
        sol = solve(new_hcdp)
        ϕ_moment_score(sol, data_moments, N)
    end
end

function A_moment_score(sol::HumanCapitalDPSolution, data_moments, N)
    start_wage_model = moments_by_period(sol, N)[1, "med_wage"]
    start_wage_data = data_moments[1, "med_wage"]
    return (start_wage_data-start_wage_model)^2
end

function A_tuner(hcdp::HumanCapitalDP, data_moments, N)
    function maximand(A)
        new_hcdp = HumanCapitalDP(hcdp.T, hcdp.k₁, hcdp.β, A, hcdp.ρ, hcdp.θ, hcdp.γ, hcdp.λ, hcdp.ϕ, hcdp.σ)
        sol = solve(new_hcdp)
        A_moment_score(sol, data_moments, N)
    end
end


function wage_hour_age_schedule_score(sol::HumanCapitalDPSolution, data_moments, N)
    model_moments = moments_by_period(sol,N)
    model_wages = model_moments[!, "med_wage"]
    data_wages = data_moments[!, "med_wage"]

    model_hours = model_moments[!, "med_hours"]
    data_hours = data_moments[!, "med_hours"]

    return sum((model_wages .- data_wages).^2) + sum((model_hours .- data_hours).^2)
end

function βρθ_tuner(hcdp::HumanCapitalDP, data_moments, N)
    function maximand(β, ρ, θ)
        new_hcdp = HumanCapitalDP(hcdp.T, hcdp.k₁, β, hcdp.A, ρ, θ, hcdp.γ, hcdp.λ, hcdp.ϕ, hcdp.σ)
        sol = solve(new_hcdp)
        wage_hour_age_schedule_score(sol, data_moments, N)
    end
end
=#