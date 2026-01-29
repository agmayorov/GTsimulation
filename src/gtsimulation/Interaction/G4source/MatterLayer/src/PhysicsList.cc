#include "PhysicsList.hh"

namespace MatterLayer
{

PhysicsList::PhysicsList()
: FTFP_BERT(0)
{
  RegisterPhysics(new G4RadioactiveDecayPhysics());
}

PhysicsList::~PhysicsList()
{}

void PhysicsList::ConstructProcess() {
    FTFP_BERT::ConstructProcess();
    // Disable neutron killing process
    G4ProcessManager* pManager = G4Neutron::Neutron()->GetProcessManager();
    G4ProcessVector* pVector = pManager->GetProcessList();
    for (G4int i = 0; i < pVector->size(); ++i) {
        G4VProcess* process = (*pVector)[i];
        if (process->GetProcessName() == "nKiller") {
            pManager->SetProcessActivation(process, false);
            break;
        }
    }
}

}
