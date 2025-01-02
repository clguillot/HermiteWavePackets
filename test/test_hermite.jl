
function test_hermite1d()

    printstyled("Testing Hermite 1d:\n"; bold=true, color=:blue)

    nb_reps = 50

    #ψₙ(a, q, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*x) * exp(-z(x - q)²/2) * exp(ipx)
    function ψ(a, q, n, x)
        (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermite_poly(n, sqrt(a) * (x-q)) * exp(-a*(x-q)^2/2)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            x = 4.0 * (rand() - 0.5)
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            val = H(x)
            val2 = dot(conj(Λ), [ψ(a, q, n, x) for n=0:N-1])
            err = max(err, abs(val - val2) / abs(val))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error evaluate (point) = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            x = SVector{5}(4.0 .* (rand(5) .- 0.5))
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            val = evaluate(H, x)
            err = max(err, norm(val - H.(x)) / norm(val))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error evaluate (vector) = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            x = 4.0 * (rand() - 0.5)
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            Hc = conj(H)

            err = max(err, abs(Hc(x) - conj(H(x))) / abs(Hc(x)))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error conjugate = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            x = 4.0 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            G = Gaussian1D(λ, a, q)
            H = convert(HermiteFct1D{N, ComplexF64, Float64, Float64}, G)

            err = max(err, abs(H(x) - G(x)) / abs(G(x)))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error conversion = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 33
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)
            
            x, w = HermiteWavePackets.hermite_quadrature(a/2, q, Val(ceil(Int, N/2)))
            I_exact = dot(w, H.(x))

            I = legendre_quadrature(20.0, 400, y -> H(y))
            err = max(err, abs(I - I_exact) / abs(I_exact))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error quadrature = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 32
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            x, M = HermiteWavePackets.hermite_discrete_transform(a, q, Val(N))
            U = M * evaluate(H, x)

            err = max(err, norm(U - Λ) / norm(Λ))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error discrete transform = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 31
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            I = legendre_quadrature(20.0, 400, y -> H(y))
            err = max(err, abs(I - integral(H)) / abs(I))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error integral = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N1 = 14
            Λ1 = (@SVector rand(N1)) + 1im * (@SVector rand(N1))
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            H1 = HermiteFct1D(Λ1, a1, q1)

            N2 = 17
            Λ2 = (@SVector rand(N2)) + 1im * (@SVector rand(N2))
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            H2 = HermiteFct1D(Λ2, a2, q2)

            H = H1 * H2

            for _=1:40
                x = 5 * (rand() - 0.5)
                err = max(err, abs(H(x) - H1(x) * H2(x)))
            end
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error product = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 17
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            Λ_P = (@SVector rand(5)) + 1im * (@SVector rand(5))
            q2 = 4 * (rand() - 0.5)

            PH = polynomial_product(q2, Λ_P, H)

            f(x) = H(x) * dot(SVector{5}((x-q2)^k for k=0:4), Λ_P)

            for _=1:10
                x = 5 * (rand() - 0.5)
                err = max(err, abs(f(x) - PH(x)) / norm_L2(PH))
            end
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error polynomial product = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N1 = 16
            Λ1 = (@SVector rand(N1)) + 1im * (@SVector rand(N1))
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            H1 = HermiteFct1D(Λ1, a1, q1)

            N2 = 14
            Λ2 = (@SVector rand(N2)) + 1im * (@SVector rand(N2))
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            H2 = HermiteFct1D(Λ2, a2, q2)

            H = convolution(H1, H2)

            x0 = 5.0 * (rand() - 0.5)
            I = legendre_quadrature(20.0, 400, y -> H1(y) * H2(x0 - y))
            err = max(err, abs(I - H(x0)))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error convolution = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N1 = 12
            Λ1 = (@SVector rand(N1)) + 1im * (@SVector rand(N1))
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            H1 = HermiteFct1D(Λ1, a1, q1)

            N2 = 19
            Λ2 = (@SVector rand(N2)) + 1im * (@SVector rand(N2))
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            H2 = HermiteFct1D(Λ2, a2, q2)

            I = legendre_quadrature(20.0, 400, y -> conj(H1(y)) * H2(y))
            err = max(err, abs(I - dot_L2(H1, H2)) / abs(I))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error dot L² = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 31
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            err = max(err, abs(norm_L2(H) - sqrt(dot_L2(H, H))) / norm_L2(H))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error norm L² = $err\n"; bold=true, color=color)
    end

    begin
        N1 = 3
        Λ1 = (@SVector rand(Float32, N1))
        a1 = (4 * rand(Float32) + 0.5f0)
        q1 = 4 * (rand(Float32) - 0.5f0)
        H1 = HermiteFct1D(Λ1, a1, q1)

        N2 = 2
        Λ2 = @SVector rand(Float32, N2)
        a2 = (4 * rand(Float32) + 0.5f0)
        q2 = 4 * (rand(Float32) - 0.5f0)
        H2 = HermiteFct1D(Λ2, a2, q2)

        N3 = 4
        Λ3 = @SVector rand(Float32, N3)
        a3 = (4 * rand(Float32) + 0.5f0)
        q3 = 4 * (rand(Float32) - 0.5f0)
        H3 = HermiteFct1D(Λ3, a3, q3)

        H = convolution(H1 * H2, H3)
        x = @SVector rand(Float32, 3)
        res = H(rand(Float32)) + integral(H) + norm_L2(H)
        T_type = promote_type(typeof(res), eltype(evaluate(H, x)))

        color = (T_type != Float32) ? :red : :green
        printstyled("Expecting $Float32 and got $T_type\n"; bold=true, color=color)
    end
end