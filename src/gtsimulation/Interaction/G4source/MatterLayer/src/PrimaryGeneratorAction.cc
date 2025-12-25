#include "PrimaryGeneratorAction.hh"

namespace MatterLayer
{

PrimaryGeneratorAction::PrimaryGeneratorAction(G4int particlePDG, G4double energy)
: G4VUserPrimaryGeneratorAction()
{
  fParticleGun = new G4ParticleGun();

  G4ParticleDefinition *particle = nullptr;
  if (G4ParticleTable::GetParticleTable()->FindParticle(particlePDG))
    particle = G4ParticleTable::GetParticleTable()->FindParticle(particlePDG);
  else if (G4IonTable::GetIonTable()->GetIon(particlePDG))
    particle = G4IonTable::GetIonTable()->GetIon(particlePDG);
  else
    G4cerr << "Error: particle was not found in G4ParticleTable and G4IonTable" << G4endl;

  fParticleGun->SetParticleDefinition(particle);
  fParticleGun->SetParticlePosition(G4ThreeVector());
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
  fParticleGun->SetParticleEnergy(energy * MeV);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent)
{
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}
