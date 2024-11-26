
function test_hermite_wave_packet1d()

    println("Testing hermite wave packet 1d:")

    nb_reps = 50

    #ψₙ(z, q, p, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*x) * exp(-z(x - q)²/2) * exp(ipx)
    function ψ(z, q, p, n, x)
        a = real(z)
        (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermite_poly(n, sqrt(a) * (x-q)) * exp(-z*(x-q)^2/2) * cis(p*x)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 55
            x = 4.0 * (rand() - 0.5)
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            val = H(x)
            val2 = dot(conj(Λ), [ψ(z, q, p, n, x) for n=0:N-1])
            err = max(err, abs(val - val2) / abs(val))
        end

        println("Error evaluate = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 17
            x = 4.0 * (rand() - 0.5)
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            Hc = conj(H)

            err = max(err, abs(Hc(x) - conj(H(x))) / abs(Hc(x)))
        end

        println("Error conjugate = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N1 = 18
            Λ1 = (@SVector rand(N1)) + 1im * (@SVector rand(N1))
            z1 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            H1 = HermiteWavePacket1D(Λ1, z1, q1, p1)

            N2 = 15
            Λ2 = (@SVector rand(N2)) + 1im * (@SVector rand(N2))
            z2 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            H2 = HermiteWavePacket1D(Λ2, z2, q2, p2)

            f(x) = H1(x) * H2(x)
            H = H1 * H2

            for j=1:100
                x = 5 * (rand() - 0.5)
                err = max(err, abs(f(x) - H(x)))
            end
        end

        println("Error product = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 17
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            I = legendre_quadrature(25.0, 500, y -> H(y))
            err = max(err, abs(I - integral(H)) / abs(I))
        end

        println("Error integral (real variance) = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 17
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5) + 1im * (0.5 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            I = legendre_quadrature(25.0, 500, y -> H(y))
            err = max(err, abs(I - integral(H)) / abs(I))
        end

        println("Error integral (complex variance) = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 17
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            Hf = fourier(H)

            ξ0 = 5 * (rand() - 0.5)
            I = legendre_quadrature(25.0, 500, y -> H(y) * cis(-ξ0*y))
            err = max(err, abs(I - Hf(ξ0)))
        end

        println("Error Fourier (real variance) = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 15
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5) + 1im * (rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            Hf = fourier(H)

            ξ0 = 5 * (rand() - 0.5)
            I = legendre_quadrature(25.0, 500, y -> H(y) * cis(-ξ0*y))
            err = max(err, abs(I - Hf(ξ0)))
        end

        println("Error Fourier (complex variance) = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 17
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            Hf = inv_fourier(H)

            ξ0 = 5 * (rand() - 0.5)
            I = 1/(2π) * legendre_quadrature(25.0, 500, y -> H(y) * cis(ξ0*y))
            err = max(err, abs(I - Hf(ξ0)))
        end

        println("Error Inverse Fourier (real variance) = $err")
    end

    begin
        err = 0.0
        for _=1:nb_reps
            N = 15
            Λ = (@SVector rand(N)) + 1im * (@SVector rand(N))
            z = (4 * rand() + 0.5) + 1im * (rand() + 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            Hf = inv_fourier(H)

            ξ0 = 5 * (rand() - 0.5)
            I = 1/(2π) * legendre_quadrature(25.0, 500, y -> H(y) * cis(ξ0*y))
            err = max(err, abs(I - Hf(ξ0)))
        end

        println("Error Inverse Fourier (complex variance) = $err")
    end

    begin
        err = 0.0

        for _=1:nb_reps
            N1 = 18
            Λ1 = (@SVector rand(N1)) + 1im * (@SVector rand(N1))
            z1 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            H1 = HermiteWavePacket1D(Λ1, z1, q1, p1)

            N2 = 15
            Λ2 = (@SVector rand(N2)) + 1im * (@SVector rand(N2))
            z2 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            H2 = HermiteWavePacket1D(Λ2, z2, q2, p2)

            H = convolution(H1, H2)

            x0 = 5.0 * (rand() - 0.5)
            I = legendre_quadrature(25.0, 500, y -> H1(y) * H2(x0 - y))
            err = max(err, abs(I - H(x0)))
        end

        println("Error convolution = $err")
    end

    begin
        err = 0.0

        for _=1:nb_reps
            N1 = 18
            Λ1 = (@SVector rand(N1)) + 1im * (@SVector rand(N1))
            z1 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            H1 = HermiteWavePacket1D(Λ1, z1, q1, p1)

            N2 = 15
            Λ2 = (@SVector rand(N2)) + 1im * (@SVector rand(N2))
            z2 = (4 * rand() + 0.5) + 1im * (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            H2 = HermiteWavePacket1D(Λ2, z2, q2, p2)

            I = legendre_quadrature(25.0, 500, y -> conj(H1(y)) * H2(y))
            err = max(err, abs(I - dot_L2(H1, H2)))
        end

        println("Error dot L² = $err")
    end

    begin
        N1 = 3
        Λ1 = (@SVector rand(Float32, N1)) + 1im * (@SVector rand(Float32, N1))
        z1 = (4 * rand(Float32) + 0.5f0) + 1im * (4 * rand(Float32) + 0.5f0)
        q1 = 4 * (rand(Float32) - 0.5f0)
        p1 = 4 * (rand(Float32) - 0.5f0)
        H1 = HermiteWavePacket1D(Λ1, z1, q1, p1)

        N2 = 7
        Λ2 = (@SVector rand(Float32, N2)) + 1im * (@SVector rand(Float32, N2))
        z2 = (4 * rand(Float32) + 0.5f0) + 1im * (4 * rand(Float32) + 0.5f0)
        q2 = 4 * (rand(Float32) - 0.5f0)
        p2 = 4 * (rand(Float32) - 0.5f0)
        H2 = HermiteWavePacket1D(Λ2, z2, q2, p2)

        N3 = 5
        Λ3 = (@SVector rand(Float32, N3)) + 1im * (@SVector rand(Float32, N3))
        z3 = (4 * rand(Float32) + 0.5f0) + 1im * (4 * rand(Float32) + 0.5f0)
        q3 = 4 * (rand(Float32) - 0.5f0)
        p3 = 4 * (rand(Float32) - 0.5f0)
        H3 = HermiteWavePacket1D(Λ3, z3, q3, p3)

        H = convolution(H1 * fourier(H2), inv_fourier(H3))
        res = H(rand(Float32)) + integral(H)
        T_type = typeof(res)

        println("Expecting $ComplexF32 and got $T_type")
    end
end