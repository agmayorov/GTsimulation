#include "PrimaryGeneratorAction.hh"

#include <G4ParticleGun.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4IonTable.hh>
#include <G4SystemOfUnits.hh>

#include <stdexcept>
#include <string>

namespace Atmosphere
{

PrimaryGeneratorAction::PrimaryGeneratorAction(const SimConfig& config)
: G4VUserPrimaryGeneratorAction(),
  fConfig(config)
{
  fParticleGun = new G4ParticleGun();
  fParticleGun->SetParticleEnergy(fConfig.energy*MeV);
  fParticleGun->SetParticlePosition(fConfig.coordinates*m);
  fParticleGun->SetParticleMomentumDirection(fConfig.direction);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  G4ParticleDefinition* particle = nullptr;
  if (G4ParticleTable::GetParticleTable()->FindParticle(fConfig.particlePDG))
    particle = G4ParticleTable::GetParticleTable()->FindParticle(fConfig.particlePDG);
  else if (G4IonTable::GetIonTable()->GetIon(fConfig.particlePDG))
    particle = G4IonTable::GetIonTable()->GetIon(fConfig.particlePDG);
  else
    throw std::invalid_argument("Particle with PDG code " + std::to_string(fConfig.particlePDG) +
                                " was not found in G4ParticleTable and G4IonTable");
  fParticleGun->SetParticleDefinition(particle);
  // Primary vertex
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}
