"""
This file implements the attack strategy described by Benjamin Wesolowski
in an unpublished note (available at https://www.dropbox.com/s/pmv3lrsg1gayl13/attacksidh.pdf)

The goal is to construct an isogeny of degree c=2^a-3^b using the
Deuring correspondence, from a quaternionic ideal of norm c.

We assume that 2^a > 3^b which is true for SIKEp217, SIKEp610 and SIKEp964.
Other cases can be handled by brute forcing the first ternary digits
(1 digit for SIKEp434, 2 for SIKEp503, 5 for SIKEp751).

We will need to following diagram to apply Kani's theorem:
- phi is the secret isogeny of degree 3^b
- aux is an auxiliary isogeny of degree c = 2^a - 3^b
  given by constructive Deuring

            phi
  Estart -------> EB
    |             |
    |             |
 aux|             |
    |             |
    v             v
    C ----------> CB

We only need to compute the image of the generators of the 2-adic and 3-adic
torsion by aux, so actual polynomials are not needed, and fractional isogeny
computations with odd denominators are allowed.

The Deuring correspondence is computed using the library available at
https://github.com/friends-of-quaternions/deuring

implementing algorithms described in:
Deuring for the People: Supersingular Elliptic Curves with
Prescribed Endomorphism Ring in General Characteristic
by Jonathan Komada Eriksen, Lorenz Panny, Jana Sotákavá, Mattia Veroni
https://eprint.iacr.org/2023/106.pdf

The library must be patched to avoid using factors 2 and 3 in ideal J.
"""

import time

from sage.all import *
from deuring.correspondence import constructive_deuring
from deuring.klpt import KLPT_Context

from helpers import supersingular_gens, fast_log3
from richelot_aux import Does22ChainSplit, Pushing3Chain
from public_values_aux import (
    generate_torsion_points,
    check_torsion_points,
    gen_bob_keypair,
)

proof.arithmetic(False)
set_verbose(1, files=["correspondence.py", "klpt.py"])

# Suitable SIKE parameters: 2^(0.9 a) < 3^b < 2^a and 2^a 3^b - 1 is prime
params = "babySIKE1"
a, b = {
    "babySIKE0": (13, 7),  # 25 bits
    "babySIKE1": (33, 19),  # 64 bits
    "babySIKE2": (51, 32),  # 102 bits
    "babySIKE3": (91, 57),  # 182 bits
    "SIKEp217": (110, 67),
    "SIKEp610": (305, 192),
}[params]
p = 2**a * 3**b - 1

# Prepare a tower of extensions, needed to evaluate quaternion elements.
Fp = GF(p)
Fp16 = GF(p**16, name="a16")
Fp4 = Fp16.subfield(4, name="a4")
Fp2 = Fp4.subfield(2, name="a2")
Fp = Fp16.subfield(1)
# Make sure coercion maps are present.
assert Fp4.coerce_map_from(Fp2)
if not Fp16.coerce_map_from(Fp2):
    print("Forcing coercion")
    a2 = Fp2.gen()
    Fp16.register_coercion(Fp2.hom([a2.minimal_polynomial().roots(ring=Fp16)[0][0]]))
    print(Fp16.coerce_map_from(Fp2))


def E_start_ext(field):
    return EllipticCurve(field, [0, 6, 0, 1, 0])


E_start = E_start_ext(Fp2)
E_start.set_order((p + 1) ** 2)


class EndoRing:
    """
    We need to determine the endomorphism ring of Estart.
    The endomorphism ring of E0: y^2 = x^3 ± x (j=1728)
    is well known: the maximal order (1, i, 1/2 + 1/2*j, 1/2*i + 1/2*k)

    There is a 2-isogeny E0 -> Estart with kernel {x=-1} (or {x=1})
    So by Deuring correspondence, it corresponds to a norm 2 ideal
    of QuaternionAlgebra(-1, -p)

    The kernel of this 2-isogeny is inside the kernel of (1-π)/2 because:
    ((1-π)/2)(P) = Q-πQ = 0 if P=2Q.

    So the ideal of the isogeny is:
    sage: I = O * 2 + O * ((1+j)/2)
    sage: I.norm()
    2
    sage: I.left_order()
    Order of Quaternion Algebra (-1, -p) with base ring Rational Field with basis (1/2 + 1/2*j, 1/2*i + 1/2*k, j, k)
    sage: I.right_order()
    Order of Quaternion Algebra (-1, -p) with base ring Rational Field with basis (1/2 + 1/2*j, 1/4*i + 1/4*k, j, 2*k)

    The endomorphsm ring of E_start is the latter maximal order, where we apply the substitution 2i => i.
    """

    def __init__(self):
        B = QuaternionAlgebra(-4, -p)
        _i, _j, _k = B.gens()
        # O is a maximal order.
        O = B.quaternion_order([1, _i, (1 + _j) / 2, (_i + _k) / 8])
        assert O.discriminant() == B.discriminant()
        self.B = B
        self.O = O

        end_i = E_start.isogeny(E_start.lift_x(Fp2(1)), codomain=E_start)
        for g in E_start.gens():
            assert end_i(end_i(g)) == -4 * g
        self.end_i = end_i

    def check(self):
        "Run checks on the implementation of the action of the endomorphism ring"
        pi = E_start.frobenius_isogeny()
        assert pi.degree() == p
        end_i = self.end_i
        # Check divisibility:
        # 1+j is divisible by 2, i+k is divisible by 8
        for t2 in E_start(0).division_points(2):
            assert t2 + pi(t2) == 0
        for t8 in E_start(0).division_points(8):
            assert end_i(t8 + pi(t8)) == 0
        # Check that the implementations are actually morphisms
        for g in E_start.gens():
            assert end_i(pi(g)) == -pi(end_i(g))
            g1 = self.end1pi2(g)
            g2 = self.endik8(g)
            for m in range(1, 5):
                print(f"Check quaternion action on {m} * {g}")
                mg = m * g
                mg1 = self.end1pi2(mg)
                assert mg1 == m * g1
                assert 2 * mg1 == mg + pi(mg)
                mg2 = self.endik8(mg)
                assert mg2 == m * g2
                assert 8 * self.endik8(mg) == end_i(mg + pi(mg))

    def end_q(self, q, P):
        "Action of an arbitrary element of O"
        O = self.O
        assert q in O
        a1, a2, a3, a4 = (
            vector(q) * Matrix(QQ, [vector(b) for b in O.basis()]).inverse()
        )
        i, j, k = self.B.gens()
        assert q == a1 + a2 * i + a3 * (1 + j) / 2 + a4 * (i + k) / 8
        p1 = a1 * P
        p2 = a2 * self.end_i(P)
        p3 = a3 * self.end1pi2(P)
        p4 = a4 * self.endik8(P)
        return p1 + p2 + p3 + p4

    def end1pi2(self, P):
        "Action of quaternion (1+j)/2"
        E2 = E_start_ext(Fp4)
        pi2 = E2.frobenius_isogeny()
        half = E2(P).division_points(2)[0]
        return E_start(half + pi2(half))

    def endik8(self, P):
        "Action of quaternion (i+k)/8"
        E8 = E_start_ext(Fp16)
        pi8 = E8.frobenius_isogeny()
        P8 = E8(*P).division_points(8)[0]
        Q = P8 + pi8(P8)
        ix, iy = self.end_i.rational_maps()
        return E_start(ix(*Q.xy()), iy(*Q.xy()))

    def solve_norm(self, c):
        # A very crude Cornacchia algorithm:
        # Select a large enough prime multiplier
        m = 2 * next_prime(isqrt(20000 * p // c))
        for w, z in ((w, z) for w in range(1, 100) for z in range(1, w)):
            # If m is even we want x and w^2+z^2 to be odd
            x = c * m**2 - p * (w**2 + z**2)
            if x > 0 and x & 3 == 1 and is_pseudoprime(x):
                # pow(2, x//2, x) == -1:
                # x looks like a 4k+1 prime
                u, v = two_squares(x)
                break
        # One of u,v and one of w,z must be even.
        if v % 2 == 1:
            u, v = v, u
        if z % 2 == 1:
            w, z = z, w
        assert c * m**2 == u**2 + v**2 + p * (w**2 + z**2)
        assert v % 2 == 0
        assert z % 2 == 0
        print(f"Quaternion {u=} {v=} {w=} {z=} has norm {c}*{m}^2")
        return u, v, w, z

    def make_ideal(self, c):
        "Select an ideal of prescribed norm c"
        B, O = self.B, self.O
        u, v, w, z = self.solve_norm(c)
        assert v % 2 == 0 and z % 2 == 0
        i, j, k = B.gens()
        assert (O * 1).left_order() == O
        assert (O * 1).right_order() == O
        q = u + (v // 2) * i + w * j + (z // 2) * k
        assert q.reduced_norm() == u * u + v * v + p * (w * w + z * z)
        assert q in O
        I = O * q + O * c
        assert c == I.norm()
        print(f"Determined ideal of norm c={factor(c)}")
        return I


endo_start = EndoRing()
# endo_start.check()

# Build an isogeny of degree c = 2^a - 3^b
# For SIKE217, c = 4357 * 11200489 * 32405783 * 762204351616943, which is not a sum of squares.
c = 2**a - 3**b
assert c > 0

I = endo_start.make_ideal(c)

# Run Deuring correspondence:
# - it requires short Weierstrsas curves
# - it must know the action of quaternion i
# - it must avoid factors 2 and 3 in the powersmooth norm

print("Starting Deuring correspondence")
EW = E_start.short_weierstrass_model()
iso_Estart_EW = E_start.isomorphism_to(EW)
K4 = iso_Estart_EW(E_start.lift_x(Fp2(1)))
assert K4.order() == 4
end_2i = EW.isogeny(K4, codomain=EW)
Eaux, phiJ, ctx = constructive_deuring(I, EW, end_2i)
print("isogeny φJ has degree", phiJ.degree())
J = ctx.J
assert J.norm() == phiJ.degree()
assert gcd(J.norm(), 6) == 1
# Now phiJ is an isogeny Ei -> C with large, powersmooth degree
# and its associated quaternionic ideal J is equivalent to original
# ideal I.
# But we want the isogeny φI of degree c.
# This is done by Algorithm 1 of https://eprint.iacr.org/2021/153.pdf
# "Evaluating non-smooth degree isogenies".
#
# We are in the case E0=E1:
# I is a connecting ideal O0 -> O2 with norm c
# J=Ix is a connecting ideal from O0 to x^-1 O2 x
#
# We need to compute I = J x^-1
# But we can instead compute I' = x^-1 J which is a different ideal,
# but it has the same norm, which is enough for us.
B = endo_start.B
O = endo_start.O
assert I.left_order() == endo_start.O
assert J.left_order() == endo_start.O
# Extract multiplier x and clear denominator
Ox = I.conjugate() * J * (1 / I.norm())
x = next(
    _g for _g in KLPT_Context(B).reducedBasis(Ox) if _g.reduced_norm() == Ox.norm()
)
alpha = x * lcm(
    _x.denominator()
    for _x in vector(x) * Matrix(QQ, [vector(b) for b in O.basis()]).inverse()
)
assert alpha in O
alpha_ratio = sqrt(J.norm() / I.norm() / alpha.reduced_norm())
assert J == I * alpha * alpha_ratio
aconj = alpha.conjugate()
ratio2 = sqrt(I.norm() / J.norm() / alpha.reduced_norm())
Ifinal = ratio2 * aconj * J
assert Ifinal.norm() == c
assert Ifinal == alpha**-1 * I * alpha
assert aconj in O


# Decompose aconj on the basis of O.
def display_a():
    a1, a2, a3, a4 = (
        vector(aconj) * Matrix(QQ, [vector(b) for b in O.basis()]).inverse()
    )
    assert aconj == sum(_a * _b for _a, _b in zip((a1, a2, a3, a4), O.basis()))
    denom = I.norm()
    print(f"Auxiliary endomorphism αbar={a1}+{a2}i + {a3}(1+j)/2 + {a4}(i+k)/8")
    print(f"Computing the final isogeny as α^-1*φJ = αbar φJ * {ratio2}")


display_a()

C = phiJ.codomain()
print("Final codomain", C)


def aux(P):
    "The final isogeny"
    # It will only be applied to 2^a torsion points.
    m = Zmod(2**a)(ratio2)
    return m * phiJ(iso_Estart_EW(endo_start.end_q(aconj, P)))


# Same code as other attacks implementations.


def run_attack(E_start, P2, Q2, P3, Q3, EB, PB, QB, aux):
    P_c = aux(P2)
    Q_c = aux(Q2)

    chain, (E1, E2) = Does22ChainSplit(C, EB, P_c, Q_c, PB, QB, a)
    print(f"Estart j={E_start.j_invariant()}")
    print(
        f"Isogeny chain splits into j-invariants {E1.j_invariant()} and {E2.j_invariant()}"
    )
    # Evaluate quotient map
    if E1.j_invariant() == E_start.j_invariant():
        index = 1
        CB = E2
    else:
        index = 0
        CB = E1

    def C_to_CB(x):
        pt = (x, None)
        for c in chain:
            pt = c(pt)
        return pt[index]

    P3_CB = C_to_CB(aux(P3))
    Q3_CB = C_to_CB(aux(Q3))

    print("Computed image of 3-adic torsion in split factor C_B")
    Z3 = Zmod(3**b)
    G1_CB, G2_CB = supersingular_gens(CB)
    G1_CB3 = ((p + 1) / 3**b) * G1_CB
    G2_CB3 = ((p + 1) / 3**b) * G2_CB
    w = G1_CB3.weil_pairing(G2_CB3, 3**b)

    sk = None
    for G in (G1_CB3, G2_CB3):
        xP = fast_log3(P3_CB.weil_pairing(G, 3**b), w)
        xQ = fast_log3(Q3_CB.weil_pairing(G, 3**b), w)
        if xQ % 3 != 0:
            sk = int(-Z3(xP) / Z3(xQ))
            break

    if sk is not None:
        # Sanity check
        bobscurve, _ = Pushing3Chain(E_start, P3 + sk * Q3, b)
        found = bobscurve.j_invariant() == EB.j_invariant()

        print(f"Bob's secret key revealed as: {sk}")
        print(f"In ternary, this is: {Integer(sk).digits(base=3)}")
        return sk
    else:
        print("Something went wrong.")
        print(f"Altogether this took {time.time() - tim} seconds.")
        return None


# Generate public torsion points, for SIKE implementations
# these are fixed but to save loading in constants we can
# just generate them on the fly
P2, Q2, P3, Q3 = generate_torsion_points(E_start, a, b)
check_torsion_points(E_start, a, b, P2, Q2, P3, Q3)

# Generate Bob's key pair
bob_private_key, EB, PB, QB = gen_bob_keypair(E_start, b, P2, Q2, P3, Q3)
solution = Integer(bob_private_key).digits(base=3)

print(
    f"Running the attack against {params} parameters, which has a prime: 2^{a}*3^{b} - 1"
)
print(f"If all goes well then the following digits should be found: {solution}")

run_attack(E_start, P2, Q2, P3, Q3, EB, PB, QB, aux)
