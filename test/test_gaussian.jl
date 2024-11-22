
function test_gaussian()

    begin
        x = 4.0 * (rand() - 0.5)
        λ = rand()
        a = (4 * rand() + 0.5)
        q = 4 * (rand() - 0.5)
        G = Gaussian1D(λ, a, q)

        println("Error evaluate = ", abs(G(x) - λ * exp(-a/2 * (x - q)^2)) / abs(λ * exp(-a/2 * (x - q)^2)))
    end

    begin
        λ1 = rand()
        a1 = (4 * rand() + 0.5)
        q1 = 4 * (rand() - 0.5)
        G1 = Gaussian1D(λ1, a1, q1)

        λ2 = rand()
        a2 = (4 * rand() + 0.5)
        q2 = 4 * (rand() - 0.5)
        G2 = Gaussian1D(λ2, a2, q2)

        f(x) = G1(x) * G2(x)
        G = G1 * G2

        err = 0.0
        for j=1:100
            x = 5 * (rand() - 0.5)
            err = max(err, abs(f(x) - G(x)) / abs(f(x)))
        end
        println("Error product = ", err)
    end

    begin
        λ1 = rand()
        a1 = (4 * rand() + 0.5)
        q1 = 4 * (rand() - 0.5)
        G1 = Gaussian1D(λ1, a1, q1)

        λ2 = rand()
        a2 = (4 * rand() + 0.5)
        q2 = 4 * (rand() - 0.5)
        G2 = Gaussian1D(λ2, a2, q2)

        G = convolution(G1, G2)

        x0 = 5.0 * (rand() - 0.5)
        N = 200
        X = 15
        h = 2 * X / N
        x_legendre, w_legendre = gausslegendre(8)
        I = 0.0
        F_conv(y) = G1(y) * G2(x0 - y)
        for k=1:N
            x = -X + (k - 0.5) * h
            I += h/2 * dot(w_legendre, F_conv.(x .+ h/2 * x_legendre))
        end
        println("Error convolution = ", abs(I - G(x0)) / abs(I))
    end

    begin
        λ = rand()
        a = (4 * rand() + 0.5)
        q = 4 * (rand() - 0.5)
        G = Gaussian1D(λ, a, q)

        N = 200
        X = 15
        h = 2 * X / N
        x_legendre, w_legendre = gausslegendre(8)
        I = 0.0
        F_int(y) = G(y)
        for k=1:N
            x = -X + (k - 0.5) * h
            I += h/2 * dot(w_legendre, F_int.(x .+ h/2 * x_legendre))
        end
        println("Error integral = ", abs(I - integral(G)) / abs(I))
    end

    begin
        λ1 = rand(Float32)
        a1 = (4 * rand(Float32) + 0.5f0)
        q1 = 4 * (rand(Float32) - 0.5f0)
        G1 = Gaussian1D(λ1, a1, q1)

        λ2 = rand(Float32)
        a2 = (4 * rand(Float32) + 0.5f0)
        q2 = 4 * (rand(Float32) - 0.5f0)
        G2 = Gaussian1D(λ2, a2, q2)

        λ3 = rand(Float32)
        a3 = (4 * rand(Float32) + 0.5f0)
        q3 = 4 * (rand(Float32) - 0.5f0)
        G3 = Gaussian1D(λ3, a3, q3)

        G = convolution(G1 * G2, G3)
        T_type = typeof(G(rand(Float32)) + integral(G))

        println("Expecting $Float32 and got $T_type")
    end
end