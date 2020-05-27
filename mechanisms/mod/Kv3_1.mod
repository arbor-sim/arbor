: Comment: Kv3-like potassium current

NEURON	{
	SUFFIX Kv3_1
	USEION k READ ek WRITE ik
	RANGE gbar, g, ik 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
	vshift = 0 (mV)
}

ASSIGNED	{
	v	(mV)
	g	(S/cm2)
	mInf
	mTau
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar*m
	ik = g*(v-ek)
}

DERIVATIVE states	{
	rates(v)
	m' = (mInf-m)/mTau
}

INITIAL{
	rates(v)
	m = mInf
}

PROCEDURE rates(v){
	UNITSOFF
		mInf =  1/(1+exp(((v -(18.700 + vshift))/(-9.700))))
		mTau =  0.2*20.000/(1+exp(((v -(-46.560 + vshift))/(-44.140))))
	UNITSON
}
