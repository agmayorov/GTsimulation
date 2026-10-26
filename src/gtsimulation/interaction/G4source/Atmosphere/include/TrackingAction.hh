#ifndef TrackingAction_hh
#define TrackingAction_hh

#include "G4UserTrackingAction.hh"
#include "G4Track.hh"

#include "G4SystemOfUnits.hh"
#include "G4VProcess.hh"
#include "RunResult.hh"

namespace Atmosphere
{

class TrackingAction : public G4UserTrackingAction
{
  public:  
    TrackingAction(RunResult& result);
   ~TrackingAction() override = default;
    void PostUserTrackingAction(const G4Track *aTrack) override;

  private:
    RunResult& fResult;
};

}

#endif
