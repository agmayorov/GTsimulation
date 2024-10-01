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
    ActionInitialization(G4int particlePDG, G4double energy, G4double height, G4double alpha);
    ~ActionInitialization() override;
    virtual void Build() const override;

  private:
    G4int fParticlePDG;
    G4double fEnergy;
    G4double fHeight;
    G4double fAlpha;
};

}

#endif
