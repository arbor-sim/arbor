module PassiveCable

using Unitful: uconvert, NoUnits

export cable_normalize, cable, rallpack1

# Compute solution g(x, t) to
#
#     ∂²g/∂x² - g - ∂g/∂t = 0
#
# on [0, L] × [0,∞), subject to:
#
#     g(x, 0) = 0
#     ∂g/∂x (0, t) = 1
#     ∂g/∂x (L, t) = 0
#
# Parameters:
#     x, t, L:  as described above
#   tol:  absolute error tolerance in result
#
# Return:
#     g(x, t)
#
# TODO: verify correctness when L≠1

function cable_normalized(x::Float64, t::Float64, L::Float64; tol=1e-8)
    if t<=0
        return 0.0
    else
        ginf = -cosh(L-x)/sinh(L)
        sum = exp(-t/L)
        Ltol = L*tol

        for k = Iterators.countfrom(1)
            a = k*pi/L
            b = exp(-t*(1+a^2))

            sum += 2*b*cos(a*x)/(1+a^2)
            resid_ub = b/(a^3*t)

            if resid_ub<Ltol
                break
            end
        end
        return ginf+sum/L;
     end
end


# Compute solution f(x, t) to
#
#     λ²∂²f/∂x² - f - τ∂f/∂t = 0
#
# on [0, L] x [0, ∞), subject to:
#
#     f(x, 0) = V
#     ∂f/∂x (0, t) = I·r
#     ∂f/∂x (L, t) = 0
#
# where:
#
#     λ² = 1/(r·g)   length constant
#     τ  = r·c       time constant
#
# In the physical model, the parameters correspond to the following:
#
#     L:  length of cable
#     r:  linear axial resistivity
#     g:  linear membrane conductivity
#     c:  linear membrane capacitance
#     V:  membrane reversal potential
#     I:  injected axial current on the left end (x = 0) of the cable.
#
# Note that r, g and c are specific 1-d quantities that differ from
# the cable resistivity r_L, specific membrane conductivity ḡ and
# specific membrane capacitance c_m as used elsewhere. If the
# cross-sectional area is A and cable circumference is f, then
# these quantities are related by:
#
#     r = r_L/A
#     g = ḡ·f
#     c = c_m·f
#
# Parameters:
#     x:  displacement along cable
#     t:  time
#     L, lambda, tau, r, V, I:  as described above
#   tol:  absolute error tolerance in result
#
# Return:
#     computed potential at (x,t) on cable.

function cable(x, t, L, lambda, tau, r, V, I; tol=1e-8)
    scale = I*r*lambda;
    if scale == 0
        return V
    else
        tol_n = abs(tol/scale)
        return scale*cable_normalized(uconvert(NoUnits, x/lambda), uconvert(NoUnits, t/tau), uconvert(NoUnits, L/lambda), tol=tol_n) + V
    end
end


# Rallpack 1 test
#
# One sided cable model with the following parameters:
#
#     RA = 1 Ω·m    bulk axial resistivity
#     RM = 4 Ω·m²   areal membrane resistivity
#     CM = 0.01 F/m²  areal membrane capacitance
#     d  = 1 µm     cable diameter
#     EM = -65 mV   reversal potential
#     I  = 0.1 nA   injected current
#     L  = 1 mm     cable length.
#
# (This notation aligns with that used in the Rallpacks paper.)
#
# Note that the injected current as described in the Rallpack model
# is trans-membrane, not axial. Consequently we need to swap the
# sign on I when passing to the cable function.
#
# Parameters:
#     x:  displacement along cable [m]
#     t:  time [s]
#   tol:  absolute tolerance for reported potential.
#
# Return:
#     computed potential at (x,t) on cable.

function rallpack1(x, t; tol=1e-8)
    RA = 1
    RM = 4
    CM = 1e-2
    d  = 1e-6
    EM = -65e-3
    I  = 0.1e-9
    L  = 1e-3

    r = 4*RA/(pi*d*d)
    lambda = sqrt(d/4 * RM/RA)
    tau = CM*RM

    return cable(x, t, L, lambda, tau, r, EM, -I, tol=tol)
end

end # module PassiveCable
