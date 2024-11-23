
function test_hermite()

    println("Testing Hermite:")

    nb_reps = 50

    #ψₙ(a, q, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*x) * exp(-z(x - q)²/2) * exp(ipx)
    function ψ(a, q, n, x)
        (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermiteh(n, sqrt(a) * (x-q)) * exp(-a*(x-q)^2/2)
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

        println("Error evaluate = $err")
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

        println("Error conjugate = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 32
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            M = 400
            X = 20
            h = 2 * X / M
            x_legendre, w_legendre = gausslegendre(8)
            I = BigFloat(0.0)
            F = y -> H(y)
            for k=1:M
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F.(x .+ h/2 * x_legendre))
            end
            I = ComplexF64(I)

            x, w = ComplexHermiteFct.hermite_quadrature(a/2, q, Val(N))
            I_exact = dot(w, F.(x))

            err = max(err, abs(I - I_exact) / abs(I_exact))
        end

        println("Error quadrature = ", err)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 32
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            x, M = ComplexHermiteFct.hermite_discrete_transform(a, q, Val(N))
            U = M * evaluate(H, x)

            err = max(err, norm(U - Λ) / norm(Λ))
        end

        println("Error discrete transform = $err")
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

        println("Error product = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 32
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = HermiteFct1D(Λ, a, q)

            M = 400
            X = 20
            h = 2 * X / M
            x_legendre, w_legendre = gausslegendre(8)
            I = BigFloat(0.0)
            F = y -> H(y)
            for k=1:M
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F.(x .+ h/2 * x_legendre))
            end
            I = ComplexF64(I)
            err = max(err, abs(I - integral(H)) / abs(I))
        end

        println("Error integral = $err")
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
            M = 400
            X = 20
            h = 2 * X / M
            x_legendre, w_legendre = gausslegendre(8)
            I = BigFloat(0.0)
            F = y -> H1(y) * H2(x0 - y)
            for k=1:M
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F.(x .+ h/2 * x_legendre))
            end
            I = ComplexF64(I)
            err = max(err, abs(I - H(x0)))
        end

        println("Error convolution = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N1 = 11
            Λ1 = (@SVector rand(N1)) + 1im * (@SVector rand(N1))
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            H1 = HermiteFct1D(Λ1, a1, q1)

            N2 = 19
            Λ2 = (@SVector rand(N2)) + 1im * (@SVector rand(N2))
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            H2 = HermiteFct1D(Λ2, a2, q2)

            M = 400
            X = 20
            h = 2 * X / M
            x_legendre, w_legendre = gausslegendre(8)
            I = BigFloat(0.0)
            F = y -> conj(H1(y)) * H2(y)
            for k=1:M
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F.(x .+ h/2 * x_legendre))
            end
            I = ComplexF64(I)
            err = max(err, abs(I - dot_L2(H1, H2)) / abs(I))
        end

        println("Error dot L² = $err")
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
        T_type = promote_type(typeof(H(rand(Float32)) + integral(H)), eltype(evaluate(H, x)))

        println("Expecting $Float32 and got $T_type")
    end
end