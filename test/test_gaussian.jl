
function test_gaussian()

    printstyled("Testing gaussian:\n"; bold=true, color=:blue)

    nb_reps = 4
    M = 12.0
    tol = 1e-12

    D = 2

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 .* (rand(D) .- 0.5)
            λ = rand() + 1im * rand()
            a = 4 .* rand(D) .+ 0.5
            q = 4 .* (rand(D) .- 0.5)
            G = Gaussian(λ, SVector{D}(a), SVector{D}(q))

            u = @. exp(-a/2 * (x - q)^2)
            μ = λ * prod(u)
            err = max(err, abs(G(x) - μ) / abs(μ))
        end

        color = (err > tol) ? :red : :green
        printstyled("Error evaluate = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            x = 4.0 .* (rand(D) .- 0.5)
            λ = rand() + 1im * rand()
            a = 4 .* rand(D) .+ 0.5
            q = 4 .* (rand(D) .- 0.5)
            G = Gaussian(λ, SVector{D}(a), SVector{D}(q))

            Gc = conj(G)

            err = max(err, abs(Gc(x) - conj(G(x))) / abs(Gc(x)))
        end

        color = (err > tol) ? :red : :green
        printstyled("Error conjugate = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            a1 = (4 .* rand(D) .+ 0.5)
            q1 = 4 .* (rand(D) .- 0.5)
            G1 = Gaussian(λ1, SVector{D}(a1), SVector{D}(q1))

            λ2 = rand() + 1im * rand()
            a2 = (4 .* rand(D) .+ 0.5)
            q2 = 4 .* (rand(D) .- 0.5)
            G2 = Gaussian(λ2, SVector{D}(a2), SVector{D}(q2))

            f(x) = G1(x) * G2(x)
            G = G1 * G2

            for j=1:100
                x = 5.0 .* (rand(D) .- 0.5)
                err = max(err, abs(f(x) - G(x)) / abs(f(x)))
            end
        end

        color = (err > tol) ? :red : :green
        printstyled("Error product = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0
        for _=1:nb_reps
            λ = rand() + 1im * rand()
            a = (4 .* rand(D) .+ 0.5)
            q = 4 .* (rand(D) .- 0.5)
            G = Gaussian(λ, SVector{D}(a), SVector{D}(q))

            I = complex_cubature(y -> G(y), [-M for _ in 1:D], [M for _ in 1:D])
            err = max(err, abs(I - integral(G)) / abs(I))
        end
        
        color = (err > tol) ? :red : :green
        printstyled("Error integral = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0

        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            a1 = (4 .* rand(D) .+ 0.5)
            q1 = 4 .* (rand(D) .- 0.5)
            G1 = Gaussian(λ1, SVector{D}(a1), SVector{D}(q1))

            λ2 = rand() + 1im * rand()
            a2 = (4 .* rand(D) .+ 0.5)
            q2 = 4 .* (rand(D) .- 0.5)
            G2 = Gaussian(λ2, SVector{D}(a2), SVector{D}(q2))

            G = convolution(G1, G2)

            x0 = 5.0 .* (rand(D) .- 0.5)
            I = complex_cubature(y -> G1(y) * G2(x0 - y), [-M for _ in 1:D], [M for _ in 1:D])
            err = max(err, abs(I - G(x0)))
        end

        color = (err > tol) ? :red : :green
        printstyled("Error convolution = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0

        for _=1:nb_reps
            λ1 = rand() + 1im * rand()
            a1 = (4 .* rand(D) .+ 0.5)
            q1 = 4 .* (rand(D) .- 0.5)
            G1 = Gaussian(λ1, SVector{D}(a1), SVector{D}(q1))

            λ2 = rand() + 1im * rand()
            a2 = (4 .* rand(D) .+ 0.5)
            q2 = 4 .* (rand(D) .- 0.5)
            G2 = Gaussian(λ2, SVector{D}(a2), SVector{D}(q2))

            I = complex_cubature(y -> conj(G1(y)) * G2(y), [-M for _ in 1:D], [M for _ in 1:D])
            err = max(err, abs(I - dot_L2(G1, G2)) / abs(I))
        end

        color = (err > tol) ? :red : :green
        printstyled("Error dot L² = $err\n"; bold=true, color=color)
    end

    begin
        err = 0.0

        for _=1:nb_reps
            λ = rand() + 1im * rand()
            a = (4 .* rand(D) .+ 0.5)
            q = 4 .* (rand(D) .- 0.5)
            G = Gaussian(λ, SVector{D}(a), SVector{D}(q))

            err = max(err, abs(norm_L2(G) - sqrt(dot_L2(G, G))) / norm_L2(G))
        end

        color = (err > tol) ? :red : :green
        printstyled("Error norm L² = $err\n"; bold=true, color=color)
    end

    begin
        λ1 = rand(Float32)
        a1 = (4 .* rand(Float32, D) .+ 0.5f0)
        q1 = 4 .* (rand(Float32, D) .- 0.5f0)
        G1 = Gaussian(λ1, SVector{D}(a1), SVector{D}(q1))

        λ2 = rand(Float32) + 1im * rand(Float32)
        a2 = (4 .* rand(Float32, D) .+ 0.5f0)
        q2 = 4 .* (rand(Float32, D) .- 0.5f0)
        G2 = Gaussian(λ2, SVector{D}(a2), SVector{D}(q2))

        λ3 = rand(Float32)
        a3 = (4 .* rand(Float32, D) .+ 0.5f0)
        q3 = 4 .* (rand(Float32, D) .- 0.5f0)
        G3 = Gaussian(λ3, SVector{D}(a3), SVector{D}(q3))


        G = convolution(G1 * G2, G3)
        res = G(rand(Float32, D)) + integral(G) + norm_L2(G)
        T_type = typeof(res)

        color = (T_type != ComplexF32) ? :red : :green
        printstyled("Expecting $ComplexF32 and got $T_type\n"; bold=true, color=color)
    end
end