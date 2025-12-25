#pragma once

#include <G4UserStackingAction.hh>

#include <G4SystemOfUnits.hh>
#include <G4EventManager.hh>

namespace MatterLayer
{

class StackingAction : public G4UserStackingAction
{
  public:
    StackingAction();
    ~StackingAction();
    G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track *aTrack) override;

  private:
    G4bool fFirstSecondaryParticle;
    G4double fDeathTime;
};

}
