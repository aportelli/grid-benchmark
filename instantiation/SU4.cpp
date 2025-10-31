#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/implementation/WilsonFermion5DImplementation.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5DImplementation.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5Dcache.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsImplementation.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsHandImplementation.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsAsmImplementation.h>
#include "SU4.hpp"

NAMESPACE_BEGIN(Grid);

template class CayleyFermion5D<SU4FundWilsonImplD>; 
template class CayleyFermion5D<SU4FundWilsonImplF>; 
template class WilsonFermion5D<SU4FundWilsonImplD>; 
template class WilsonFermion5D<SU4FundWilsonImplF>; 
template class WilsonKernels<SU4FundWilsonImplD>;
template class WilsonKernels<SU4FundWilsonImplF>;

NAMESPACE_END(Grid);
