
function test_hermite()

    printstyled("Testing hermite functions:\n"; bold=true, color=:blue)

    nb_reps = 1
    M = 10.0
    tol = 1e-6
    int_tol = 1e-9

    #ψₙ(a, q, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*x) * exp(-z(x - q)²/2) * exp(ipx)
    function ψ(a, q, n, x)
        (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermite_poly(n, sqrt(a) * (x-q)) * exp(-a*(x-q)^2/2)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 6)
        alloc = 0
        for _=1:nb_reps
            x = 4.0 .* (rand(D) .- 0.5)
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = 4 .* rand(D) .+ 0.5
            q = 4 .* (rand(D) .- 0.5)
            H = HermiteFct(Λ, SVector{D}(a), SVector{D}(q))

            alloc += @allocated μ1 = H(x)
            μ2 = sum(Λ[j...] * prod(ψ(H.z[k], H.q[k], j[k]-1, x[k]) for k in 1:D) for j in Iterators.product(axes(Λ)...))
            err = max(err, abs(μ1 - μ2) / abs(μ1))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error evaluate = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (3, 5, 4)
        Ngrid = Tuple{3, 6, 5}
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = 4 .* rand(D) .+ 0.5
            q = 4 .* (rand(D) .- 0.5)
            H = HermiteFct(Λ, SVector{D}(a), SVector{D}(q))

            agrid = SVector{D}(4 .* rand(D) .+ 0.5)
            qgrid = SVector{D}(4 .* (rand(D) .- 0.5))
            xgrid = hermite_grid(agrid, qgrid, Ngrid)

            alloc += @allocated A = evaluate_grid(H, xgrid)

            for j in Iterators.product(eachindex.(xgrid)...)
                x = SVector((xgrid[k][j[k]] for k in eachindex(xgrid))...)
                μ1 = H(x)
                μ2 = A[j...]
                err = max(err, abs(μ1 - μ2) / abs(μ1))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error evaluate grid = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end
    
    let
        err = 0.0
        D = 3
        N = (5, 4, 6)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            xgrid = hermite_grid(a, q, Tuple{N...})
            alloc += @allocated U = hermite_discrete_transform(evaluate_grid(H, xgrid), a)

            err = max(err, norm(U - Λ) / norm(Λ))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error discrete transform = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 2
        N = (5, 3)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            alloc += @allocated μ1 = integral(H)
            μ2 = complex_cubature(y -> H(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error integral = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N1 = (5, 4, 6)
        N2 = (3, 5, 7)
        alloc = 0
        for _=1:nb_reps
            Λ1 = SArray{Tuple{N1...}}(rand(N1...) + im * rand(N1...))
            a1 = SVector{D}(4 .* rand(D) .+ 0.5)
            q1 = SVector{D}(4 .* (rand(D) .- 0.5))
            H1 = HermiteFct(Λ1, a1, q1)

            Λ2 = SArray{Tuple{N2...}}(rand(N2...) + im * rand(N2...))
            a2 = SVector{D}(4 .* rand(D) .+ 0.5)
            q2 = SVector{D}(4 .* (rand(D) .- 0.5))
            H2 = HermiteFct(Λ2, a2, q2)

            alloc += @allocated H = H1 * H2

            for _=1:40
                x = 5 .* (SVector{D}(rand(D)) .- 0.5)
                err = max(err, abs(H(x) - H1(x) * H2(x)))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error product = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 6)
        NP = (3, 7, 5)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            Λ_P = (@SArray rand(NP...)) + 1im * (@SArray rand(NP...))
            q2 = SVector{D}(4 .* (rand(D) .- 0.5))

            alloc += @allocated PH = polynomial_product(H, Λ_P, q2)

            f(x) = H(x) * dot(SArray{Tuple{NP...}}(prod((x-q2).^(k.-1); init=1.0) for k in Iterators.product((1:NP[j] for j in 1:D)...)), Λ_P)

            for _=1:10
                x = 5 .* (SVector{D}(rand(D)) .- 0.5)
                err = max(err, abs(f(x) - PH(x)) / norm_L2(PH))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error polynomial product = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 2
        N = (6, 5)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            alloc += @allocated HF = fourier(H)
            ξ = SVector{D}(4.0 .* (rand(D) .- 0.5))
            μ1 = HF(ξ)
            μ2 = complex_cubature(y -> H(y) * cis(-dot(y, ξ)), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error Fourier = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 7)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)
            HF = fourier(H)

            alloc += @allocated H_ = inv_fourier(HF)
            x = SVector{D}(4.0 .* (rand(D) .- 0.5))
            μ1 = H_(x)
            μ2 = H(x)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error inverse Fourier = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 2
        N1 = (5, 2)
        N2 = (3, 4)
        alloc = 0
        for _=1:nb_reps
            Λ1 = SArray{Tuple{N1...}}(rand(N1...) + im * rand(N1...))
            a1 = SVector{D}(4 .* rand(D) .+ 0.5)
            q1 = SVector{D}(4 .* (rand(D) .- 0.5))
            H1 = HermiteFct(Λ1, a1, q1)

            Λ2 = SArray{Tuple{N2...}}(rand(N2...) + im * rand(N2...))
            a2 = SVector{D}(4 .* rand(D) .+ 0.5)
            q2 = SVector{D}(4 .* (rand(D) .- 0.5))
            H2 = HermiteFct(Λ2, a2, q2)

            alloc += @allocated μ1 = dot_L2(H1, H2)
            μ2 = complex_cubature(y -> conj(H1(y)) * H2(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error dot L2 = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 7)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            a = SVector{D}(4 .* rand(D) .+ 0.5)
            q = SVector{D}(4 .* (rand(D) .- 0.5))
            H = HermiteFct(Λ, a, q)

            alloc += @allocated μ1 = norm_L2(H)
            μ2 = sqrt(dot_L2(H, H))
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error norm_L2 = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        D = 3
        N1 = (3, 2, 6)
        N2 = (4, 7, 6)
        N3 = (3, 2, 4)

        Λ1 = SArray{Tuple{N1...}}(rand(Float32, N1...))
        a1 = SVector{D}(4 .* rand(Float32, D) .+ 0.5f0)
        q1 = SVector{D}(4 .* (rand(Float32, D) .- 0.5f0))
        H1 = HermiteFct(Λ1, a1, q1)

        Λ2 = SArray{Tuple{N2...}}(rand(Float32, N2...))
        a2 = SVector{D}(4 .* rand(Float32, D) .+ 0.5f0)
        q2 = SVector{D}(4 .* (rand(Float32, D) .- 0.5f0))
        H2 = HermiteFct(Λ2, a2, q2)

        Λ3 = SArray{Tuple{N3...}}(rand(Float32, N3...))
        a3 = SVector{D}(4 .* rand(Float32, D) .+ 0.5f0)
        q3 = SVector{D}(4 .* (rand(Float32, D) .- 0.5f0))
        H3 = HermiteFct(Λ3, a3, q3)

        H = convolution(H1 * H2, H3)
        res = H(rand(Float32, D)) + integral(H) + norm_L2(H)
        T_type = typeof(res)

        color = (T_type != Float32) ? :red : :green
        printstyled("Expecting $Float32 and got $T_type\n"; bold=true, color=color)
    end
end