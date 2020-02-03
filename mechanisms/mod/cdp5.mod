: Calcium ion accumulation with endogenous buffers, DCM and pump

: Changes:
: function argument names
: kinetic <<
: f_flux/b_flux (not needed)

:COMMENT
:
:The basic code of Example 9.8 and Example 9.9 from NEURON book was adapted as:
:
:1) Extended using parameters from Schmidt et al. 2003.
:2) Pump rate was tuned according to data from Maeda et al. 1999
:3) DCM was introduced and tuned to approximate the effect of radial diffusion
:
:Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
:
:*Article available as Open Access
:
:PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
:
:Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
:Contact: Haroon Anwar (anwar@oist.jp)
:
:ENDCOMMENT


NEURON {
    SUFFIX cdp5
    USEION ca READ cao, ica WRITE cai
    RANGE ica_pmp
    RANGE Nannuli, Buffnull2, rf3, rf4, vrat
    RANGE TotalPump
}


UNITS {
    (mol)   = (1)
    (molar) = (1/liter)
    (mM)    = (millimolar)
    (um)    = (micron)
    (mA)    = (milliamp)
:    FARADAY = (faraday)  (10000 coulomb)
:    PI      = (pi)       (1)
}

PARAMETER {
    diam              (um)

    Nannuli = 10.9495 (1)
    celsius           (degC)

    cainull = 45e-6   (mM)
    mginull = 0.59    (mM)

:    values for a buffer compensating the diffusion
    Buffnull1 = 0         (mM)
    rf1       = 0.0134329 (/ms mM)
    rf2       = 0.0397469 (/ms)

    Buffnull2 = 60.9091   (mM)
    rf3       = 0.1435    (/ms mM)
    rf4       = 0.0014    (/ms)

:    values for benzothiazole coumarin (BTC)
    BTCnull   = 0         (mM)
    b1        = 5.33      (/ms mM)
    b2        = 0.08      (/ms)

:    values for caged compound DMNPE-4
    DMNPEnull = 0         (mM)
    c1        = 5.63      (/ms mM)
    c2        = 0.107e-3  (/ms)

:   values for Calbindin (2 high and 2 low affinity binding sites)
    CBnull    = .16       (mM)
    nf1       = 43.5      (/ms mM)
    nf2       = 3.58e-2   (/ms)
    ns1       = 5.5       (/ms mM)
    ns2       = 0.26e-2   (/ms)

:   values for Parvalbumin
    PVnull    = .08       (mM)
    m1        = 1.07e2    (/ms mM)
    m2        = 9.5e-4    (/ms)
    p1        = 0.8       (/ms mM)
    p2        = 2.5e-2    (/ms)

    kpmp1     = 3e-3      (/mMms)
    kpmp2     = 1.75e-5   (/ms)
    kpmp3     = 7.255e-5  (/ms)
    TotalPump = 1e-9      (mol/cm2)

:    cai :neuron
}

ASSIGNED {
    ica_pmp   (mA/cm2)
    parea     (um)     : pump area per unit length
    parea2      (um)
    mgi          (mM)
    vrat      (1)

:    ica :neuron
}

CONSTANT {
    FARADAY = 96.485309
    R       = 8.313424
    PI      = 3.1415927

}

STATE {
    : ca[0] is equivalent to cai
    : ca[] are very small, so specify absolute tolerance
    : let it be ~1.5 - 2 orders of magnitude smaller than baseline level

    cai                   :magic, needs to be here

    ca            (mM)    :<1e-3>
    mg            (mM)    :<1e-6>

    Buff1         (mM)
    Buff1_ca      (mM)

    Buff2         (mM)
    Buff2_ca      (mM)

    BTC           (mM)
    BTC_ca        (mM)

    DMNPE         (mM)
    DMNPE_ca      (mM)

    CB            (mM)
    CB_f_ca       (mM)
    CB_ca_s       (mM)
    CB_ca_ca      (mM)

    PV            (mM)
    PV_ca         (mM)
    PV_mg         (mM)

    pump          (mol/cm2) :<1e-15>
    pumpca        (mol/cm2) :<1e-15>
}

BREAKPOINT {
    SOLVE state METHOD sparse
    cai = ca
}

INITIAL {
    factors()

    ca = cainull
    mg = mginull

    Buff1    = ssBuff1()
    Buff1_ca = ssBuff1ca()

    Buff2    = ssBuff2()
    Buff2_ca = ssBuff2ca()

    BTC      = ssBTC()
    BTC_ca   = ssBTCca()

    DMNPE    = ssDMNPE()
    DMNPE_ca = ssDMNPEca()

    CB       = ssCB    ( kdf(), kds())
    CB_f_ca  = ssCBfast( kdf(), kds())
    CB_ca_s  = ssCBslow( kdf(), kds())
    CB_ca_ca = ssCBca  ( kdf(), kds())

    PV       = ssPV    ( kdc(), kdm())
    PV_ca    = ssPVca  ( kdc(), kdm())
    PV_mg    = ssPVmg  ( kdc(), kdm())

    parea  = PI*diam
    parea2 = PI*(diam-0.2)

:    ica_pmp_last = 0
    ica     = 0
    ica_pmp = 0
    pump    = TotalPump
    pumpca  = 0

    cai = ca
}

PROCEDURE factors() {
    LOCAL r, dr2
    r    = 1/2                 : starts at edge (half diam)
    dr2  = r/(Nannuli-1)/2     : full thickness of outermost annulus,
    vrat = PI*(r-dr2/2)*2*dr2  : interior half
}

KINETIC state {
    LOCAL dsq, dsqvol

    COMPARTMENT diam*diam*vrat {ca Buff1 Buff1_ca Buff2 Buff2_ca BTC BTC_ca DMNPE DMNPE_ca CB CB_f_ca CB_ca_s CB_ca_ca PV PV_ca}
    COMPARTMENT (1e10)*parea   {pump pumpca}

    :pump
    ~ ca + pump <-> pumpca  ( kpmp1*parea*(1e10), kpmp2*parea*(1e10) )
    ~    pumpca <-> pump    ( kpmp3*parea*(1e10), 0 )

    CONSERVE pump + pumpca = TotalPump * parea * (1e10)

    :all currents except pump
    :ica is Ca efflux
    :ica_pmp = 2*FARADAY*(f_flux - b_flux)/parea    : Is this used anywhere?

    ~ ca <-> (0, -ica*PI*diam/(2*FARADAY)) :arbor
:    ~ ca << (-ica*PI*diam/(2*FARADAY)) :neuron

    :RADIAL DIFFUSION OF ca, mg and mobile buffers
    dsq    = diam*diam
    dsqvol = dsq*vrat

    ~ ca + Buff1 <-> Buff1_ca   ( rf1*dsqvol, rf2*dsqvol )
    ~ ca + Buff2 <-> Buff2_ca   ( rf3*dsqvol, rf4*dsqvol )
    ~ ca + BTC   <-> BTC_ca     ( b1 *dsqvol,  b2*dsqvol )
    ~ ca + DMNPE <-> DMNPE_ca   ( c1 *dsqvol,  c2*dsqvol )

    :Calbindin
    ~ ca + CB      <-> CB_ca_s  ( nf1*dsqvol, nf2*dsqvol )
    ~ ca + CB      <-> CB_f_ca  ( ns1*dsqvol, ns2*dsqvol )
    ~ ca + CB_f_ca <-> CB_ca_ca ( nf1*dsqvol, nf2*dsqvol )
    ~ ca + CB_ca_s <-> CB_ca_ca ( ns1*dsqvol, ns2*dsqvol )

    :Paravalbumin
    ~ ca + PV <-> PV_ca         ( m1*dsqvol,   m2*dsqvol )
}

FUNCTION ssBuff1() {
    ssBuff1 = Buffnull1/(1+((rf1/rf2)*cainull))
}
FUNCTION ssBuff1ca() {
    ssBuff1ca = Buffnull1/(1+(rf2/(rf1*cainull)))
}
FUNCTION ssBuff2() {
    ssBuff2 = Buffnull2/(1+((rf3/rf4)*cainull))
}
FUNCTION ssBuff2ca() {
    ssBuff2ca = Buffnull2/(1+(rf4/(rf3*cainull)))
}

FUNCTION ssBTC() {
    ssBTC = BTCnull/(1+((b1/b2)*cainull))
}

FUNCTION ssBTCca() {
    ssBTCca = BTCnull/(1+(b2/(b1*cainull)))
}

FUNCTION ssDMNPE() {
    ssDMNPE = DMNPEnull/(1+((c1/c2)*cainull))
}

FUNCTION ssDMNPEca() {
    ssDMNPEca = DMNPEnull/(1+(c2/(c1*cainull)))
}

FUNCTION ssCB( k0, k1) {
    ssCB = CBnull/(1+k0+k1+(k0*k1))
}

FUNCTION ssCBfast( k0, k1) {
    ssCBfast = (CBnull*k1)/(1+k0+k1+(k0*k1))
}

FUNCTION ssCBslow( k0, k1) {
    ssCBslow = (CBnull*k0)/(1+k0+k1+(k0*k1))
}

FUNCTION ssCBca(k0, k1) {
    ssCBca = (CBnull*k0*k1)/(1+k0+k1+(k0*k1))
}

FUNCTION kdf() {
    kdf = (cainull*nf1)/nf2
}

FUNCTION kds() {
    kds = (cainull*ns1)/ns2
}

FUNCTION kdc() {
    kdc = (cainull*m1)/m2
}

FUNCTION kdm() {
    kdm = (mginull*p1)/p2
}

FUNCTION ssPV( k0, k1 ) {
    ssPV = PVnull/(1+k0+k1)
}

FUNCTION ssPVca( k0, k1 ) {
    ssPVca = (PVnull*k0)/(1+k0+k1)
}

FUNCTION ssPVmg( k0, k1) {
    ssPVmg = (PVnull*k1)/(1+k0+k1)
}