import matplotlib.pyplot as plt

def sobol_2d(n_points):
    """
    Generate first `n_points` of a 2D Sobol sequence
    using the polynomial x^3 + x^2 + x + 1 for dimension 2.
    We'll do integer XORs and then divide by 2^32 at the end.
    """
    max_bits = 32  # enough bits for typical use

    # ----------------------------------------------------------------
    # 1) Direction numbers for dimension 1 (simple):
    #    Let v1[j] = 1 << (32 - j).  That means the j-th direction
    #    corresponds to a "1" in the (32-j)-th bit position.
    # ----------------------------------------------------------------
    v1 = [1 << (32 - j) for j in range(1, max_bits+1)]

    # ----------------------------------------------------------------
    # 2) Direction numbers for dimension 2, from polynomial x^3 + x^2 + x + 1.
    #    We are given m1=1, m2=3, m3=5, then for j>3:
    #        m_j = (m_{j-1} << 1) ^ (m_{j-2} << 2) ^ (m_{j-3} << 3)
    #    Then define v2[j] = m_j << (32 - j).
    # ----------------------------------------------------------------
    m = [0, 1, 3, 5]  # index 0 unused; m[1]=1, m[2]=3, m[3]=5
    for j in range(4, max_bits+1):
        mj = (m[j-1] << 1) ^ (m[j-2] << 2) ^ (m[j-3] << 3)
        m.append(mj)
    v2 = [m[j] << (32 - j) for j in range(1, max_bits+1)]

    # ----------------------------------------------------------------
    # 3) Gray-code iteration.  Keep x1_int and x2_int as 32-bit
    #    integers.  Then x1 = x1_int / 2^32 is the float in [0,1).
    # ----------------------------------------------------------------
    points = []
    x1_int = 0
    x2_int = 0

    for i in range(n_points):
        # Convert current integer coordinates to float in [0,1)
        x1 = x1_int / float(1 << 32)
        x2 = x2_int / float(1 << 32)
        points.append((x1, x2))

        # Find rightmost set bit of (i+1) to pick direction
        idx = (i+1) & -(i+1)
        L = idx.bit_length()  # 1-based index of that set bit

        # XOR with the L-th direction number
        x1_int ^= v1[L-1]
        x2_int ^= v2[L-1]

    return points


if __name__ == "__main__":
    n = 50
    sobol_points = sobol_2d(n)

    x_coords = [p[0] for p in sobol_points]
    y_coords = [p[1] for p in sobol_points]

    plt.figure(figsize=(6,6))
    plt.scatter(x_coords, y_coords, marker="o", color="blue")
    plt.title(f"First {n} points of 2D Sobol sequence")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    plt.savefig("sobol_sequence.png")