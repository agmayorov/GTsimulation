#ifndef PhysicsList_hh
#define PhysicsList_hh

#include "FTFP_BERT.hh"

#include "G4SystemOfUnits.hh"
#include "G4RadioactiveDecayPhysics.hh"

namespace Atmosphere
{

class PhysicsList : public FTFP_BERT
{
  public:
    PhysicsList();
    ~PhysicsList();
};

}

#endif
