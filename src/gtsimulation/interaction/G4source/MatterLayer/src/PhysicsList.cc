#include "PhysicsList.hh"

#include <G4RadioactiveDecayPhysics.hh>
#include <G4Neutron.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessVector.hh>
#include <G4VProcess.hh>

namespace MatterLayer
{

PhysicsList::PhysicsList()
: FTFP_BERT(0)
{
  RegisterPhysics(new G4RadioactiveDecayPhysics());
}

void PhysicsList::ConstructProcess()
{
  FTFP_BERT::ConstructProcess();
  // Remove neutron killing process
  G4ProcessManager* pManager = G4Neutron::Neutron()->GetProcessManager();
  G4ProcessVector* pVector = pManager->GetProcessList();
  for (G4int i = 0; i < pVector->size(); ++i) {
    G4VProcess* process = (*pVector)[i];
    if (process->GetProcessName() == "nKiller") {
      pManager->RemoveProcess(process);
      break;
    }
  }
}

}
