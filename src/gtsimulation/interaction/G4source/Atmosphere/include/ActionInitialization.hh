#pragma once

#include <G4VUserActionInitialization.hh>
#include "SimConfig.hh"
#include "RunResult.hh"

namespace Atmosphere
{

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(const SimConfig& config, RunResult& result);
    ~ActionInitialization() override = default;
    void Build() const override;

  private:
    const SimConfig& fConfig;
    RunResult& fResult;
};

}
