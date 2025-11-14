#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/implementation/WilsonFermion5DImplementation.h>
#include "SU4.hpp"

NAMESPACE_BEGIN(Grid);

template class WilsonFermion5D<SU4FundWilsonImplD>;

NAMESPACE_END(Grid);
