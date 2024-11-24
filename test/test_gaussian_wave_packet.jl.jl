
function test_gaussian_wave_packet1d()

    println("Testing gaussian wave packet 1d:")

    nb_reps = 13

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

            μ = λ * exp(-z/2 * (x - q)^2) * cis(p*x)
            err = max(err, abs(G(x) - μ) / abs(μ))
        end

        println("Error evaluate = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)
            
            Gc = conj(G)

            err = max(err, abs(Gc(x) - conj(G(x))) / abs(Gc(x)))
        end

        println("Error conjugate = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            z1 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            G1 = GaussianWavePacket1D(λ1, z1, q1, p1)

            λ2 = rand() + 1im * rand()
            z2 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            G2 = GaussianWavePacket1D(λ2, z2, q2, p2)

            f(x) = G1(x) * G2(x)
            G = G1 * G2

            for j=1:100
                x = 5 * (rand() - 0.5)
                err = max(err, abs(f(x) - G(x)) / abs(f(x)))
            end
        end

        println("Error product = $err")
    end

    begin
        err = 0.0

        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            z1 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            G1 = GaussianWavePacket1D(λ1, z1, q1, p1)

            λ2 = rand() + 1im * rand()
            z2 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            G2 = GaussianWavePacket1D(λ2, z2, q2, p2)

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

            err = max(err, abs(I - G(x0)))
        end

        println("Error convolution = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

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
            
            err = max(err, abs(I - integral(G)) / abs(I))
        end

        println("Error integral = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            ξ = 5 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

            Gf = fourier(G)

            N = 200
            X = 15
            h = 2 * X / N
            x_legendre, w_legendre = gausslegendre(8)
            I = 0.0
            F_int(y) = G(y) * exp(-1im * ξ * y)
            for k=1:N
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F_int.(x .+ h/2 * x_legendre))
            end
            
            err = max(err, abs(I - Gf(ξ)) / abs(I))
        end

        println("Error Fourier = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            ξ = 5 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

            Gf = inv_fourier(G)

            N = 200
            X = 15
            h = 2 * X / N
            x_legendre, w_legendre = gausslegendre(8)
            I = 0.0
            F_int(y) = (2π)^(-1) * G(y) * exp(1im * ξ * y)
            for k=1:N
                x = -X + (k - 0.5) * h
                I += h/2 * dot(w_legendre, F_int.(x .+ h/2 * x_legendre))
            end
            
            err = max(err, abs(I - Gf(ξ)) / abs(I))
        end

        println("Error Inverse Fourier = $err")
    end

    begin
        λ1 = rand(Float32) + 1im * rand(Float32)
        z1 = (4 * rand(Float32) + 0.5f0) + 1im * (4 * rand(Float32) + 0.5f0)
        q1 = 4 * (rand(Float32) - 0.5f0)
        p1 = 4 * (rand(Float32) - 0.5f0)
        G1 = GaussianWavePacket1D(λ1, z1, q1, p1)

        λ2 = rand(Float32) + 1im * rand(Float32)
        z2 = (4 * rand(Float32) + 0.5f0) + 1im * (4 * rand(Float32) + 0.5f0)
        q2 = 4 * (rand(Float32) - 0.5f0)
        p2 = 4 * (rand(Float32) - 0.5f0)
        G2 = GaussianWavePacket1D(λ2, z2, q2, p2)

        λ3 = rand(Float32) + 1im * rand(Float32)
        z3 = (4 * rand(Float32) + 0.5f0) + 1im * (4 * rand(Float32) + 0.5f0)
        q3 = 4 * (rand(Float32) - 0.5f0)
        p3 = 4 * (rand(Float32) - 0.5f0)
        G3 = GaussianWavePacket1D(λ3, z3, q3, p3)

        G = convolution(G1 * fourier(G2), inv_fourier(G3))
        res = G(rand(Float32)) + integral(G)
        T_type = typeof(res)

        println("Expecting $ComplexF32 and got $T_type")
    end
end