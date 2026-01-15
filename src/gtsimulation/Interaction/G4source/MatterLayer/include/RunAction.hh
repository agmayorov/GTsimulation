#pragma once

#include <G4UserRunAction.hh>
#include <vector>

namespace MatterLayer
{

struct ParticleData {
  std::string name;
  int pdgCode;
  double mass;
  double charge;
  double kineticEnergy;
  std::vector<double> momentumDirection; // [x, y, z]
  std::vector<double> position;          // [x, y, z]
  std::string lastProcess;
};

struct RunResult {
  ParticleData primary;
  std::vector<ParticleData> secondaries;
};

class RunAction : public G4UserRunAction
{
  public:
    RunAction();
    ~RunAction();
    void BeginOfRunAction(const G4Run* aRun) override;

    static void SetPrimaryParticle(ParticleData data);
    static void AddSecondaryParticle(ParticleData data);
    static const RunResult& GetResult();

  private:
    static RunResult fCurrentResult;
};

}
