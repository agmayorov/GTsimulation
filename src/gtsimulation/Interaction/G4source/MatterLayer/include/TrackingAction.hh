#pragma once

#include <G4UserTrackingAction.hh>

#include <G4SystemOfUnits.hh>
#include <G4VProcess.hh>

namespace MatterLayer
{

class TrackingAction : public G4UserTrackingAction
{
  public:  
    TrackingAction();
    ~TrackingAction();
    void PostUserTrackingAction(const G4Track *aTrack) override;
};

}
