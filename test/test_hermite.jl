
function test_hermite()

    nb_reps = 5

    #ψₙ(a, q, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*x) * exp(-z(x - q)²/2) * exp(ipx)
    function ψ(a, q, n, x)
        (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermiteh(n, sqrt(a) * (x-q)) * exp(-a*(x-q)^2/2)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            x = 4.0 * (rand() - 0.5)
            Λ = @SVector rand(N)
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = StaticHermiteFct1D(Λ, a, q)

            val = H(x)
            val2 = dot(Λ, [ψ(a, q, n, x) for n=0:N-1])
            err = max(err, abs(val - val2) / abs(val))
        end
        println("Error evaluate = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            Λ = @SVector rand(N)
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = StaticHermiteFct1D(Λ, a, q)

            M = 300
            X = 16
            h = 2 * X / M
            x_legendre, w_legendre = gausslegendre(8)
            I = 0.0
            F = x -> H(x)
            for k=1:M
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F.(x .+ h/2 * x_legendre))
            end

            x, w = hermite_quadrature(a/2, q, Val(N))
            I_exact = dot(w, F.(x))

            err = max(err, abs(I - I_exact) / abs(I_exact))
        end

        println("Error quadrature = ", err)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            Λ = @SVector rand(N)
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = StaticHermiteFct1D(Λ, a, q)

            x, M = hermite_discrete_transform(a, q, Val(N))
            U = M * evaluate(H, x)

            err = max(err, norm(U - Λ) / norm(Λ))
        end

        println("Error discrete transform = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N1 = 64
            Λ1 = @SVector rand(N1)
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            H1 = StaticHermiteFct1D(Λ1, a1, q1)

            N2 = 50
            Λ2 = @SVector rand(N2)
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            H2 = StaticHermiteFct1D(Λ2, a2, q2)

            H = H1 * H2
            F_prod = x -> H1(x) * H2(x)

            for j=1:20
                x = 5 * (rand() - 0.5)
                err = max(err, abs(H(x) - F_prod(x)))
            end
        end

        println("Error product = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 16
            Λ = @SVector rand(N)
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            H = StaticHermiteFct1D(Λ, a, q)

            M = 300
            X = 16
            h = 2 * X / M
            x_legendre, w_legendre = gausslegendre(8)
            I = 0.0
            F_int = x -> H(x)
            for k=1:M
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F_int.(x .+ h/2 * x_legendre))
            end
            err = max(err, abs(I - integral(H)) / abs(I))
        end

        println("Error integral = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N1 = 4
            Λ1 = @SVector rand(N1)
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            H1 = StaticHermiteFct1D(Λ1, a1, q1)

            N2 = 3
            Λ2 = @SVector rand(N2)
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            H2 = StaticHermiteFct1D(Λ2, a2, q2)

            H = convolution(H1, H2)

            x0 = 5.0 * (rand() - 0.5)
            M = 300
            X = 16
            h = 2 * X / M
            x_legendre, w_legendre = gausslegendre(8)
            I = 0.0
            F = y -> H1(y) * H2(x0 - y)
            for k=1:M
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F.(x .+ h/2 * x_legendre))
            end
            err = max(err, abs(I - H(x0)))
        end

        println("Error convolution = $err")
    end
end