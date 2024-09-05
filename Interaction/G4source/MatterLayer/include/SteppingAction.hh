#ifndef SteppingAction_hh
#define SteppingAction_hh

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"

#include "G4VProcess.hh"

namespace MatterLayer
{

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction();
    ~SteppingAction();
    void UserSteppingAction(const G4Step *step) override;

  private:
    G4ThreeVector fPointOfInteraction;
    G4ThreeVector fDeltaVector;
};

}

#endif
