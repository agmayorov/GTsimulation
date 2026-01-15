#include "ActionInitialization.hh"

namespace MatterLayer
{

ActionInitialization::ActionInitialization(const SimConfig* config)
: fConfig(config)
{}

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fConfig));
  SetUserAction(new RunAction());
  SetUserAction(new StackingAction());
  SetUserAction(new TrackingAction());
}

}
