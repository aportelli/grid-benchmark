#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsImplementation.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsHandImplementation.h>
#include <Grid/qcd/action/fermion/implementation/WilsonKernelsAsmImplementation.h>
#include "SU4.hpp"

NAMESPACE_BEGIN(Grid);

template class WilsonKernels<SU4FundWilsonImplF>;

NAMESPACE_END(Grid);
