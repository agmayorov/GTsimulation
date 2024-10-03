#ifndef TrackingAction_hh
#define TrackingAction_hh

#include "G4UserTrackingAction.hh"
#include "G4Track.hh"

#include "G4SystemOfUnits.hh"
#include "G4VProcess.hh"

namespace Atmosphere
{

class TrackingAction : public G4UserTrackingAction
{
  public:  
    TrackingAction();
   ~TrackingAction() override = default;
    void  PreUserTrackingAction(const G4Track *aTrack) override;
    void PostUserTrackingAction(const G4Track *aTrack) override;

  private:
    G4bool fFirstSecondaryParticle;
};

}

#endif
