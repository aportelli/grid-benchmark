#pragma once

#include <Grid/Grid.h>


NAMESPACE_BEGIN(Grid);

typedef WilsonImpl<Grid::vComplexF, 
                   Grid::FundamentalRep<4,Grid::GroupName::SU>, 
                   Grid::CoeffReal> SU4FundWilsonImplF;
typedef WilsonImpl<Grid::vComplexD, 
                   Grid::FundamentalRep<4,Grid::GroupName::SU>, 
                   Grid::CoeffReal> SU4FundWilsonImplD;

NAMESPACE_END(Grid);
