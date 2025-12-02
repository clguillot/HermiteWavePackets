
function test_hermite_wave_packets()

    printstyled("Testing hermite wave packets:\n"; bold=true, color=:blue)

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
            x = 4.0 * (rand(D) .- 0.5)
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)

            alloc += @allocated μ1 = H(x)
            μ2 = sum(Λ[j...] * prod(ψ(real(H.z[k]), H.q[k], j[k]-1, x[k]) for k in 1:D) for j in Iterators.product(axes(Λ)...)) *
                    exp(sum(-im * imag(z) / 2 .* (x - q).^2) + im * dot(x, p))
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
            z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)

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
            λ = rand() + 1im * rand()
            z = SVector{D}((4 * rand(D) .+ 0.5) + 4im * (rand(D) .- 0.5))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            G = GaussianWavePacket(λ, Diagonal(z), q, p)

            alloc += @allocated H = HermiteWavePacket(G)
            alloc += @allocated G_ = truncate_to_gaussian(H)

            for _ in 1:100
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                err = max(err, abs(H(x) - G(x)))
                err = max(err, abs(G_(x) - G(x)))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error convert from gaussian = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 2
        N = (5, 3)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)

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
            z1 = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q1 = SVector{D}(4 * (rand(D) .- 0.5))
            p1 = SVector{D}(4 * (rand(D) .- 0.5))
            H1 = HermiteWavePacket(Λ1, z1, q1, p1)

            Λ2 = SArray{Tuple{N2...}}(rand(N2...) + im * rand(N2...))
            z2 = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q2 = SVector{D}(4 * (rand(D) .- 0.5))
            p2 = SVector{D}(4 * (rand(D) .- 0.5))
            H2 = HermiteWavePacket(Λ2, z2, q2, p2)

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
            z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)

            b = SVector{D}(4 * (rand(D) .- 0.5))
            q2 = SVector{D}(4 * (rand(D) .- 0.5))
            p2 = SVector{D}(4 * (rand(D) .- 0.5))

            f(x) = H(x) * cis(-dot(x - q2, Diagonal(b/2), x - q2)) * cis(dot(p2, x))
            alloc += @allocated H2 = unitary_product(H, Diagonal(b), q2, p2)

            for _=1:100
                x = SVector{D}(5 * (rand(D) .- 0.5))
                err = max(err, abs(f(x) - H2(x)))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error unitary product = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 6)
        NP = (3, 7, 5)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)

            Λ_P = SArray{Tuple{NP...}}(rand(NP...) + im * rand(NP...))
            q2 = SVector{D}(4 .* (rand(D) .- 0.5))

            alloc += @allocated PH = polynomial_product(H, Λ_P, q2)

            f(x) = H(x) * dot(SArray{Tuple{NP...}}(prod((x-q2).^(k.-1); init=1.0) for k in Iterators.product((1:NP[j] for j in 1:D)...)), Λ_P)

            for _=1:100
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
            z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)

            alloc += @allocated HF = fourier(H)

            ξ = SVector{D}(4.0 .* (rand(D) .- 0.5))
            μ1 = HF(ξ)
            μ2 = complex_cubature(y -> H(y) * cis(-dot(y, ξ)), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error Fourier (complex variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 2
        N = (6, 5)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            z = SVector{D}(4 * rand(D) .+ 0.5)
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)
            Hc = HermiteWavePacket(Λ, complex.(z), q, p)

            alloc += @allocated HF = fourier(H)
            HFc = fourier(Hc)

            for _ in 1:100
                ξ = SVector{D}(4.0 .* (rand(D) .- 0.5))
                μ1 = HF(ξ)
                μ2 = HFc(ξ)
                err = max(err, abs(μ1 - μ2))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error Fourier (real variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 7)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)
            HF = fourier(H)

            alloc += @allocated H_ = inv_fourier(HF)
            for _ in 1:100
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                μ1 = H_(x)
                μ2 = H(x)
                err = max(err, abs(μ1 - μ2))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error inverse Fourier (complex variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 3
        N = (5, 4, 7)
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            z = SVector{D}(4 * rand(D) .+ 0.5)
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)
            HF = fourier(H)

            alloc += @allocated H_ = inv_fourier(HF)
            for _ in 1:100
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                μ1 = H_(x)
                μ2 = H(x)
                err = max(err, abs(μ1 - μ2))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error inverse Fourier (real variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        D = 2
        N1 = (5, 2)
        N2 = (3, 4)
        alloc = 0
        for _=1:nb_reps
            Λ1 = SArray{Tuple{N1...}}(rand(N1...) + im * rand(N1...))
            z1 = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q1 = SVector{D}(4 * (rand(D) .- 0.5))
            p1 = SVector{D}(4 * (rand(D) .- 0.5))
            H1 = HermiteWavePacket(Λ1, z1, q1, p1)

            Λ2 = SArray{Tuple{N2...}}(rand(N2...) + im * rand(N2...))
            z2 = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
            q2 = SVector{D}(4 * (rand(D) .- 0.5))
            p2 = SVector{D}(4 * (rand(D) .- 0.5))
            H2 = HermiteWavePacket(Λ2, z2, q2, p2)

            alloc += @allocated μ1 = dot_L2(H1, H2)
            μ2 = complex_cubature(y -> conj(H1(y)) * H2(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error dot L2 = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        D = 3
        N = (5, 9, 10)
        err = 0.0
        alloc = 0
        for _=1:nb_reps
            Λ = SArray{Tuple{N...}}(rand(N...) + im * rand(N...))
            z = SVector{D}(4 * rand(D) .+ 0.5)
            q = SVector{D}(4 * (rand(D) .- 0.5))
            p = SVector{D}(4 * (rand(D) .- 0.5))
            H = HermiteWavePacket(Λ, z, q, p)

            alloc += @allocated μ1 = norm_L2(H)
            μ2 = sqrt(dot_L2(H, H))
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error norm L² = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        D = 3
        N1 = (3, 2, 6)
        N2 = (4, 7, 6)
        N3 = (3, 2, 4)

        Λ1 = SArray{Tuple{N1...}}(rand(Float32, N1...) + im * rand(Float32, N1...))
        z1 = SVector{D}(4 * rand(Float32, D) .+ 0.5f0 + im * (4 * rand(Float32, D) .- 2))
        q1 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
        p1 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
        H1 = HermiteWavePacket(Λ1, z1, q1, p1)

        Λ2 = SArray{Tuple{N2...}}(rand(Float32, N2...) + im * rand(Float32, N2...))
        z2 = SVector{D}(4 * rand(Float32, D) .+ 0.5f0 + im * (4 * rand(Float32, D) .- 2))
        q2 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
        p2 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
        H2 = HermiteWavePacket(Λ2, z2, q2, p2)

        Λ3 = SArray{Tuple{N1...}}(rand(Float32, N1...) + im * rand(Float32, N1...))
        z3 = SVector{D}(4 * rand(Float32, D) .+ 0.5f0 + im * (4 * rand(Float32, D) .- 2))
        q3 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
        p3 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
        H3 = HermiteWavePacket(Λ3, z3, q3, p3)

        H = convolution(H1 * H2, H3)
        res = H(rand(Float32, D)) + integral(H) + norm_L2(H)
        T_type = typeof(res)

        color = (T_type != ComplexF32) ? :red : :green
        printstyled("Expecting $ComplexF32 and got $T_type\n"; bold=true, color=color)
    end
end