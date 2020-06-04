: Based on Im model of Vervaeke et al. (2006)

NEURON	{
	SUFFIX Im_v2
	USEION k READ ek WRITE ik
	RANGE gbar, g, ik
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	celsius         (degC)     
	gbar = 0.00001 (S/cm2)
}

ASSIGNED	{
	v	(mV)
	g	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar * m
	ik = g * (v - ek)
}

DERIVATIVE states	{
	rates(v, celsius)
	m' = (mInf - m) / mTau
}

INITIAL{
	rates(v, celsius)
	m = mInf
}

PROCEDURE rates(v, celsius) {
  LOCAL qt
  qt = 2.3^((celsius-30)/10)
  mAlpha = 0.007 * exp( (6 * 0.4 * (v - (-48))) / 26.12 )
  mBeta = 0.007 * exp( (-6 * (1 - 0.4) * (v - (-48))) / 26.12 )

	mInf = mAlpha / (mAlpha + mBeta)
  mTau = (15 + 1 / (mAlpha + mBeta)) / qt
}
