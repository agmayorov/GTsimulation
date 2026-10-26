#ifndef PhysicsList_hh
#define PhysicsList_hh

#include "FTFP_BERT.hh"

#include "G4SystemOfUnits.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4Neutron.hh"
#include "G4ProcessManager.hh"
#include "G4VProcess.hh"

namespace Atmosphere
{

class PhysicsList : public FTFP_BERT
{
  public:
    PhysicsList();
    ~PhysicsList();
    virtual void ConstructProcess() override;
};

}

#endif
