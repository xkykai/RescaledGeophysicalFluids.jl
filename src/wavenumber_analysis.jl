function calculate_critical_k(bs, xb)
    N = length(bs)
    Δx = xb[2] - xb[1]

    F = rfft(bs)
    ks = rfftfreq(N, 1/Δx)

    max_k_ind = argmax(abs.(F[2:end])) + 1
    kc = 2π * (ks[max_k_ind])
    kc_neighbourhood = 2π .* ks[max_k_ind-1:max_k_ind+1]
    return kc, kc_neighbourhood
end