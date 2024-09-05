#ifndef ActionInitialization_hh
#define ActionInitialization_hh

#include "G4VUserActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"
#include "TrackingAction.hh"

namespace DecayGenerator
{

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(G4int particlePDG, G4double energy);
    ~ActionInitialization() override;
    virtual void Build() const override;

  private:
    G4int fParticlePDG;
    G4double fEnergy;
};

}

#endif
