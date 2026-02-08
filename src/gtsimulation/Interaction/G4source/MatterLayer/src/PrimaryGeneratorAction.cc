#include "PrimaryGeneratorAction.hh"

namespace MatterLayer
{

PrimaryGeneratorAction::PrimaryGeneratorAction(const SimConfig* config)
: fConfig(config)
{
  fParticleGun = new G4ParticleGun();
  fParticleGun->SetParticlePosition(G4ThreeVector());
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  // Particle type
  G4ParticleDefinition* particle = nullptr;
  if (G4ParticleTable::GetParticleTable()->FindParticle(fConfig->particlePDG))
    particle = G4ParticleTable::GetParticleTable()->FindParticle(fConfig->particlePDG);
  else if (G4IonTable::GetIonTable()->GetIon(fConfig->particlePDG))
    particle = G4IonTable::GetIonTable()->GetIon(fConfig->particlePDG);
  else
    std::cerr << "Error: Particle was not found in G4ParticleTable and G4IonTable" << std::endl;
  fParticleGun->SetParticleDefinition(particle);
  // Particle energy
  fParticleGun->SetParticleEnergy(fConfig->energy * MeV);
  // Primary vertex
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}
