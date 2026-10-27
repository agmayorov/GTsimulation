#pragma once

#include <G4UserTrackingAction.hh>
#include "RunResult.hh"

namespace Atmosphere
{

class TrackingAction : public G4UserTrackingAction
{
  public:
    TrackingAction(RunResult& result);
    ~TrackingAction() override = default;
    void PostUserTrackingAction(const G4Track* aTrack) override;

  private:
    RunResult& fResult;
};

}
