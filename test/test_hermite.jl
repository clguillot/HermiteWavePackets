
function test_hermite()

    printstyled("Testing hermite functions:\n"; bold=true, color=:blue)

    nb_reps = 1
    M = 10.0
    tol = 5e-12
    int_tol = 1e-13

    #ψₙ(a, q, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*x) * exp(-z(x - q)²/2) * exp(ipx)
    function ψ(a, q, n, x)
        (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermite_poly(n, sqrt(a) * (x-q)) * exp(-a*(x-q)^2/2)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 6)
        for _=1:nb_reps
            x = 4.0 .* (rand(D) .- 0.5)
            Λ = (@SArray rand(N...)) + im * (@SArray rand(N...))
            a = 4 .* rand(D) .+ 0.5
            q = 4 .* (rand(D) .- 0.5)
            H = HermiteFct(Λ, SVector{D}(a), SVector{D}(q))

            @time μ1 = H(x)
            μ2 = sum(Λ[j...] * prod(ψ(H.z[k], H.q[k], j[k]-1, x[k]) for k in 1:D) for j in Iterators.product(axes(Λ)...))
            err = max(err, abs(μ1 - μ2) / abs(μ1))
        end

        color = (err > tol) ? :red : :green
        printstyled("Error evaluate = $err\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (3, 5, 4)
        Ngrid = Tuple{3, 6, 5}
        for _=1:nb_reps
            Λ = (@SArray rand(N...)) + im * (@SArray rand(N...))
            a = 4 .* rand(D) .+ 0.5
            q = 4 .* (rand(D) .- 0.5)
            H = HermiteFct(Λ, SVector{D}(a), SVector{D}(q))

            agrid = SVector{D}(4 .* rand(D) .+ 0.5)
            qgrid = SVector{D}(4 .* (rand(D) .- 0.5))
            xgrid = hermite_grid(agrid, qgrid, Ngrid)

            @time A = evaluate_grid(H, xgrid)

            for j in Iterators.product(eachindex.(xgrid)...)
                x = SVector((xgrid[k][j[k]] for k in eachindex(xgrid))...)
                μ1 = H(x)
                μ2 = A[j...]
                err = max(err, abs(μ1 - μ2) / abs(μ1))
            end
        end

        color = (err > tol) ? :red : :green
        printstyled("Error evaluate grid = $err\n"; bold=true, color=color)
    end
    
    let
        err = 0.0
        D = 3
        N = (5, 4, 6)
        for _=1:nb_reps
            Λ = (@SArray rand(N...)) + im * (@SArray rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            xgrid = hermite_grid(a, q, Tuple{N...})
            @time U = hermite_discrete_transform(evaluate_grid(H, xgrid), a)

            err = max(err, norm(U - Λ) / norm(Λ))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error discrete transform = $err\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 1
        N = (5)
        for _=1:nb_reps
            Λ = (@SArray rand(N...)) + im * (@SArray rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            @time μ1 = integral(H)
            μ2 = complex_cubature(y -> H(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error integral = $err\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N1 = (5, 4, 6)
        N2 = (3, 5, 7)
        for _=1:nb_reps
            Λ1 = (@SArray rand(N1...)) + 1im * (@SArray rand(N1...))
            a1 = SVector{D}(4 .* rand(D) .+ 0.5)
            q1 = SVector{D}(4 .* (rand(D) .- 0.5))
            H1 = HermiteFct(Λ1, a1, q1)

            Λ2 = (@SArray rand(N2...)) + 1im * (@SArray rand(N2...))
            a2 = SVector{D}(4 .* rand(D) .+ 0.5)
            q2 = SVector{D}(4 .* (rand(D) .- 0.5))
            H2 = HermiteFct(Λ2, a2, q2)

            @time H = H1 * H2

            for _=1:40
                x = 5 .* (SVector{D}(rand(D)) .- 0.5)
                err = max(err, abs(H(x) - H1(x) * H2(x)))
            end
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error product = $err\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 6)
        NP = (3, 7, 5)
        for _=1:nb_reps
            Λ = (@SArray rand(N...)) + im * (@SArray rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            Λ_P = (@SArray rand(NP...)) + 1im * (@SArray rand(NP...))
            q2 = SVector{D}(4 .* (rand(D) .- 0.5))

            @time PH = polynomial_product(q2, Λ_P, H)

            f(x) = H(x) * dot(SArray{Tuple{NP...}}(prod((x-q2).^(k.-1); init=1.0) for k in Iterators.product((1:NP[j] for j in 1:D)...)), Λ_P)

            for _=1:10
                x = 5 .* (SVector{D}(rand(D)) .- 0.5)
                err = max(err, abs(f(x) - PH(x)) / norm_L2(PH))
            end
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error polynomial product = $err\n"; bold=true, color=color)
    end
end