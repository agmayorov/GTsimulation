#pragma once

#include <G4UserTrackingAction.hh>

namespace MatterLayer
{

class TrackingAction : public G4UserTrackingAction
{
  public:
    TrackingAction() = default;
    ~TrackingAction() override = default;
    void PostUserTrackingAction(const G4Track* aTrack) override;
};

}
