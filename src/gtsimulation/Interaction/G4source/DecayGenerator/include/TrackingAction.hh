#ifndef TrackingAction_hh
#define TrackingAction_hh

#include "G4UserTrackingAction.hh"
#include "G4Track.hh"

namespace DecayGenerator
{

class TrackingAction : public G4UserTrackingAction
{
  public:  
    TrackingAction();
   ~TrackingAction() override = default;
   
    void  PreUserTrackingAction(const G4Track *aTrack) override;
    void PostUserTrackingAction(const G4Track *aTrack) override;
};

}

#endif
