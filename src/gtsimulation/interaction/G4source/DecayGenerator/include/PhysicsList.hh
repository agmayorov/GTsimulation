#ifndef PhysicsList_hh
#define PhysicsList_hh

#include "G4VModularPhysicsList.hh"

#include "G4SystemOfUnits.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"

namespace DecayGenerator
{

class PhysicsList : public G4VModularPhysicsList
{
  public:
    PhysicsList();
    ~PhysicsList();
};

}

#endif
