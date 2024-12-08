
function test_gaussian1d()

    printstyled("Testing gaussian 1d:\n"; bold=true, color=:blue)

    nb_reps = 13

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            G = Gaussian1D(λ, a, q)

            err = max(err, abs(G(x) - λ * exp(-a/2 * (x - q)^2)) / abs(λ * exp(-a/2 * (x - q)^2)))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error evaluate = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 * (rand() - 0.5)
            λ = rand() + 1im * rand()
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            G = Gaussian1D(λ, a, q)

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
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            G1 = Gaussian1D(λ1, a1, q1)

            λ2 = rand() + 1im * rand()
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            G2 = Gaussian1D(λ2, a2, q2)

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
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            G = Gaussian1D(λ, a, q)

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
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            G1 = Gaussian1D(λ1, a1, q1)

            λ2 = rand() + 1im * rand()
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            G2 = Gaussian1D(λ2, a2, q2)

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
            a1 = (4 * rand() + 0.5)
            q1 = 4 * (rand() - 0.5)
            G1 = Gaussian1D(λ1, a1, q1)

            λ2 = rand() + 1im * rand()
            a2 = (4 * rand() + 0.5)
            q2 = 4 * (rand() - 0.5)
            G2 = Gaussian1D(λ2, a2, q2)

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
            a = (4 * rand() + 0.5)
            q = 4 * (rand() - 0.5)
            G = Gaussian1D(λ, a, q)

            err = max(err, abs(norm_L2(G) - sqrt(dot_L2(G, G))) / norm_L2(G))
        end

        color = (err > 5e-13) ? :red : :green
        printstyled("Error norm L² = $err\n"; bold=true, color=color)
    end

    begin
        λ1 = rand(Float32)
        a1 = (4 * rand(Float32) + 0.5f0)
        q1 = 4 * (rand(Float32) - 0.5f0)
        G1 = Gaussian1D(λ1, a1, q1)

        λ2 = rand(Float32) + 1im * rand(Float32)
        a2 = (4 * rand(Float32) + 0.5f0)
        q2 = 4 * (rand(Float32) - 0.5f0)
        G2 = Gaussian1D(λ2, a2, q2)

        λ3 = rand(Float32)
        a3 = (4 * rand(Float32) + 0.5f0)
        q3 = 4 * (rand(Float32) - 0.5f0)
        G3 = Gaussian1D(λ3, a3, q3)

        G = convolution(G1 * G2, G3)
        res = G(rand(Float32)) + integral(G) + norm_L2(G)
        T_type = typeof(res)

        color = (T_type != ComplexF32) ? :red : :green
        printstyled("Expecting $ComplexF32 and got $T_type\n"; bold=true, color=color)
    end
end