#pragma once

#include <G4UserStackingAction.hh>
#include <G4Types.hh>
#include "SimConfig.hh"

namespace MatterLayer
{

class StackingAction : public G4UserStackingAction
{
  public:
    StackingAction(const SimConfig& config);
    ~StackingAction() override = default;
    G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* aTrack) override;
    void PrepareNewEvent() override;

  private:
    G4double fDeathTime;
    const SimConfig& fConfig;
};

}
