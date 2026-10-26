#include "TrackingAction.hh"

namespace Atmosphere
{

TrackingAction::TrackingAction(RunResult& result)
: fResult(result)
{}

void TrackingAction::PostUserTrackingAction(const G4Track *aTrack)
{
  G4bool isPrimary = aTrack->GetTrackID() == 1;
  G4bool isLeavingWorld = aTrack->GetTrackID() != 1 && aTrack->GetKineticEnergy() > 1. &&
                          aTrack->GetStep()->GetPostStepPoint()->GetStepStatus() == fWorldBoundary;
  if (!isPrimary && !isLeavingWorld)
    return;

  G4ThreeVector momentumDirection = aTrack->GetMomentumDirection();
  G4ThreeVector position = aTrack->GetPosition() / m;
  G4ThreeVector vertexMomentumDirection = aTrack->GetVertexMomentumDirection();
  G4ThreeVector vertexPosition = aTrack->GetVertexPosition() / m;
  ParticleData data {
    aTrack->GetParticleDefinition()->GetParticleName(),
    aTrack->GetParticleDefinition()->GetPDGEncoding(),
    aTrack->GetParticleDefinition()->GetPDGMass(),
    aTrack->GetParticleDefinition()->GetPDGCharge(),
    aTrack->GetKineticEnergy() / MeV,
    std::vector<double> {momentumDirection.x(), momentumDirection.y(), momentumDirection.z()},
    std::vector<double> {position.x(), position.y(), position.z()},
    aTrack->GetVertexKineticEnergy() / MeV,
    std::vector<double> {vertexMomentumDirection.x(), vertexMomentumDirection.y(), vertexMomentumDirection.z()},
    std::vector<double> {vertexPosition.x(), vertexPosition.y(), vertexPosition.z()},
    aTrack->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()
  };

  if (isPrimary)
    fResult.primary = data;
  else
    fResult.secondaries.push_back(data);
}

}
