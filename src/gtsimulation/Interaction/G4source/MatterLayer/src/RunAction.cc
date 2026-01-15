#include "RunAction.hh"

namespace MatterLayer
{

RunResult RunAction::fCurrentResult;

RunAction::RunAction()
{}

RunAction::~RunAction()
{}

void RunAction::BeginOfRunAction(const G4Run* aRun) {
  fCurrentResult.secondaries.clear();
}

void RunAction::SetPrimaryParticle(ParticleData data) {
  fCurrentResult.primary = data;
}

void RunAction::AddSecondaryParticle(ParticleData data) {
  fCurrentResult.secondaries.push_back(data);
}

const RunResult& RunAction::GetResult() {
  return fCurrentResult;
}

}
