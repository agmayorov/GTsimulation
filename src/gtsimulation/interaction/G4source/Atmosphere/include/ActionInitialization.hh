#ifndef ActionInitialization_hh
#define ActionInitialization_hh

#include "G4VUserActionInitialization.hh"
#include "SimConfig.hh"
#include "RunResult.hh"

#include "PrimaryGeneratorAction.hh"
#include "TrackingAction.hh"

namespace Atmosphere
{

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(const SimConfig& config, RunResult& result);
    ~ActionInitialization() override;
    virtual void Build() const override;

  private:
    const SimConfig& fConfig;
    RunResult& fResult;
};

}

#endif
