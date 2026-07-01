#include "TrackingAction.hh"

namespace DecayGenerator
{

TrackingAction::TrackingAction()
{}

void TrackingAction::PreUserTrackingAction(const G4Track *aTrack)
{}

void TrackingAction::PostUserTrackingAction(const G4Track *aTrack)
{
  std::cout.precision(16);
  if (aTrack->GetTrackID() == 1 && aTrack->GetStep()->GetNumberOfSecondariesInCurrentStep() > 0)
  {
    G4cout << "Information about the secondary particles:\n"
           << "Name,PDGcode,Mass[MeV],Charge,LifeTime[ns],KineticEnergy[MeV],MomentumDirection" << G4endl;
  }
  if (aTrack->GetTrackID() != 1)
  {
    std::cout << aTrack->GetParticleDefinition()->GetParticleName() << ","
              << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
              << aTrack->GetParticleDefinition()->GetPDGMass() << ","
              << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
              << aTrack->GetParticleDefinition()->GetPDGLifeTime() << ","
              << aTrack->GetKineticEnergy() << ","
              << aTrack->GetMomentumDirection() << std::endl;
  }
}

}
