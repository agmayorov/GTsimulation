#include "G4RunManager.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

#ifdef USE_VISUALIZATION
  #include "G4UImanager.hh"
  #include "G4UIExecutive.hh"
  #include "G4VisExecutive.hh"
#endif

using namespace DecayGenerator;

int main(int argc, char* argv[])
{
  // Read input
  // Input example: ./DecayGenerator 0 15 500
  if (argc != 4) {
    G4cerr << "Wrong number of input parameters" << G4endl;
    return 3;
  }
  // Read values of input variables
  G4long seed       = atol(argv[1]);
  G4int particlePDG = atoi(argv[2]); // PDG code of particle
  G4double energy   = atof(argv[3]); // MeV
  G4cout << "Input particlePDG: " << particlePDG << "\n"
         << "Input energy: " << energy << " MeV" << "\n\n"
         << "Seed: " << seed << "\n" << G4endl;

  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine);
  CLHEP::HepRandom::setTheSeed(seed);

  // Construct RunManager and initialize G4 kernel
  G4RunManager *runManager = new G4RunManager();
  runManager->SetUserInitialization(new DetectorConstruction());
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(particlePDG, energy));
  runManager->Initialize();

  #ifdef USE_VISUALIZATION
    // Get the pointer to the User Interface manager
    G4UImanager *UImanager = G4UImanager::GetUIpointer();
    G4VisManager *visManager = new G4VisExecutive();
    visManager->Initialize();
    G4UIExecutive *UI = new G4UIExecutive(argc, argv);
    UImanager->ApplyCommand("/control/execute vis.mac");
    UI->SessionStart();
    delete UI;
  #else
    // Run 1 particle
    runManager->BeamOn(1);
  #endif

  // Job termination
  delete runManager;

  return 0;
}
