#pragma once

#include <G4VUserActionInitialization.hh>

#include "PrimaryGeneratorAction.hh"
#include "StackingAction.hh"
#include "TrackingAction.hh"

namespace MatterLayer
{

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(G4int particlePDG, G4double energy);
    ~ActionInitialization();
    void Build() const override;

  private:
    G4int fParticlePDG;
    G4double fEnergy;
};

}
