#ifndef ActionInitialization_hh
#define ActionInitialization_hh

#include "G4VUserActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"
#include "TrackingAction.hh"

namespace Atmosphere
{

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(G4int particlePDG, G4double energy, G4ThreeVector coordinates, G4ThreeVector velocity);
    ~ActionInitialization() override;
    virtual void Build() const override;

  private:
    G4int fParticlePDG;
    G4double fEnergy;
    G4ThreeVector fCoordinates;
    G4ThreeVector fVelocity;
};

}

#endif
