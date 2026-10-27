#include "ActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "StackingAction.hh"
#include "TrackingAction.hh"

namespace MatterLayer
{

ActionInitialization::ActionInitialization(const SimConfig& config)
: fConfig(config)
{}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fConfig));
  SetUserAction(new RunAction());
  SetUserAction(new StackingAction(fConfig));
  SetUserAction(new TrackingAction());
}

}
