#ifndef PrimaryGeneratorAction_hh
#define PrimaryGeneratorAction_hh

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"

#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"

namespace Atmosphere
{

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction(G4int particlePDG, G4double energy, G4double height, G4double alpha);
    ~PrimaryGeneratorAction() override;
    virtual void GeneratePrimaries(G4Event *anEvent) override;

  private:
    G4ParticleGun *fParticleGun;
    G4int fParticlePDG;
    G4double fEnergy;
    G4double fHeight;
    G4double fAlpha;
};

}

#endif
