
function test_gaussian_wave_packet1d()

    printstyled("Testing gaussian wave packet 1d:\n"; bold=true, color=:blue)

    nb_reps = 13

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

            μ = λ * exp(-z/2 * (x - q)^2) * cis(p*x)
            err = max(err, abs(G(x) - μ) / abs(μ))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error evaluate = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)
            
            Gc = conj(G)

            err = max(err, abs(Gc(x) - conj(G(x))) / abs(Gc(x)))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error conjugate = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            z1 = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            G1 = GaussianWavePacket1D(λ1, z1, q1, p1)

            λ2 = rand() + 1im * rand()
            z2 = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
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

        color = (err > 5e-13) ? :red : :green
        printstyled("Error product = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

            b = 4 * (rand() - 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)

            f2(x) = G(x) * cis(-b/2 * (x - q2)^2) * cis(p2*x)
            G2 = unitary_product(b, q2, p2, G)
            f3(x) = G(x) * cis(-b/2 * x^2)
            G3 = unitary_product(b, G)

            for j=1:100
                x = 5 * (rand() - 0.5)
                err = max(err, abs(f2(x) - G2(x)))
                err = max(err, abs(f3(x) - G3(x)))
            end
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error unitary product = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

            I = legendre_quadrature(15.0, 200, y -> G(y))            
            err = max(err, abs(I - integral(G)) / abs(I))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error integral = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0

        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            z1 = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            G1 = GaussianWavePacket1D(λ1, z1, q1, p1)

            λ2 = rand() + 1im * rand()
            z2 = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            G2 = GaussianWavePacket1D(λ2, z2, q2, p2)

            G = convolution(G1, G2)

            x0 = 5.0 * (rand() - 0.5)
            I = legendre_quadrature(15.0, 200, y -> G1(y) * G2(x0 - y))
            err = max(err, abs(I - G(x0)))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error convolution = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0

        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            z1 = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            G1 = GaussianWavePacket1D(λ1, z1, q1, p1)

            λ2 = rand() + 1im * rand()
            z2 = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            G2 = GaussianWavePacket1D(λ2, z2, q2, p2)

            I = legendre_quadrature(15.0, 200, y -> conj(G1(y)) * G2(y))
            err = max(err, abs(I - dot_L2(G1, G2)) / abs(I))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error dot L² = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0

        for _=1:nb_reps
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket1D(λ, z, q, p)

            err = max(err, abs(norm_L2(G) - sqrt(dot_L2(G, G))) / norm_L2(G))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error norm L² = $err\n"; bold=true, color=color)
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

            I = legendre_quadrature(15.0, 200, y -> G(y) * exp(-1im * ξ * y))            
            err = max(err, abs(I - Gf(ξ)) / abs(I))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error Fourier = $err\n"; bold=true, color=color)
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

            I = (2π)^(-1) * legendre_quadrature(15.0, 200, y -> G(y) * cis(ξ * y))
            err = max(err, abs(I - Gf(ξ)) / abs(I))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error Inverse Fourier = $err\n"; bold=true, color=color)
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
        res = G(rand(Float32)) + integral(G) + norm_L2(G)
        T_type = typeof(res)

        color = (T_type != ComplexF32) ? :red : :green
        printstyled("Expecting $ComplexF32 and got $T_type\n"; bold=true, color=color)
    end
end