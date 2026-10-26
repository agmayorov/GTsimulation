#include "PrimaryGeneratorAction.hh"

namespace Atmosphere
{

PrimaryGeneratorAction::PrimaryGeneratorAction(const SimConfig& config)
: G4VUserPrimaryGeneratorAction(),
  fParticleGun(0),
  fConfig(config)
{
  fParticleGun = new G4ParticleGun();
  fParticleGun->SetParticleEnergy(fConfig.energy*MeV);
  fParticleGun->SetParticlePosition(fConfig.coordinates*km);
  fParticleGun->SetParticleMomentumDirection(fConfig.direction);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent)
{
  G4ParticleDefinition *particle = 0;
  if (G4ParticleTable::GetParticleTable()->FindParticle(fConfig.particlePDG))
    particle = G4ParticleTable::GetParticleTable()->FindParticle(fConfig.particlePDG);
  else if (G4IonTable::GetIonTable()->GetIon(fConfig.particlePDG))
    particle = G4IonTable::GetIonTable()->GetIon(fConfig.particlePDG);
  else
    std::cerr << "Error: particle was not found in G4ParticleTable and G4IonTable" << std::endl;
  fParticleGun->SetParticleDefinition(particle);
  // Primary vertex
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}
