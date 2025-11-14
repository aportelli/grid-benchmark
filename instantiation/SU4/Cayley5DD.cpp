#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5DImplementation.h>
#include <Grid/qcd/action/fermion/implementation/CayleyFermion5Dcache.h>
#include "SU4.hpp"

NAMESPACE_BEGIN(Grid);

template class CayleyFermion5D<SU4FundWilsonImplD>;

NAMESPACE_END(Grid);
