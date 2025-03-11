#include "G4RunManager.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

#include "G4UImanager.hh"

#ifdef USE_VISUALIZATION
  #include "G4UIExecutive.hh"
  #include "G4VisExecutive.hh"
#endif

using namespace MatterLayer;

int main(int argc, char* argv[])
{
  // Read input
  // Input example: ./MatterLayer 0 2212 500 60 0.001 0 0 0.75 0.25 0
  if (argc != 11) {
    G4cout << "Wrong number of input parameters" << G4endl;
    return 0;
  }
  // Read values of input variables
  G4long seed = atol(argv[1]);
  G4int particlePDG = atoi(argv[2]); // PDG code of particle
  G4double energy   = atof(argv[3]); // MeV
  G4double mass     = atof(argv[4]); // g/cm^2
  G4double density  = atof(argv[5]); // g/cm^3
  G4double w_H  = atof(argv[6]);
  G4double w_He = atof(argv[7]);
  G4double w_N  = atof(argv[8]);
  G4double w_O  = atof(argv[9]);
  G4double w_Ar = atof(argv[10]);
  G4cout << "Input particlePDG: " << particlePDG << "\n"
         << "Input energy: " << energy << " MeV" << "\n"
         << "Input mass: " << mass << " g/cm2" << "\n"
         << "Input density: " << density << " g/cm3" << "\n\n"
         << "Input w_H: "  << w_H  << "\n"
         << "Input w_He: " << w_He << "\n"
         << "Input w_N: "  << w_N  << "\n"
         << "Input w_O: "  << w_O  << "\n"
         << "Input w_Ar: " << w_Ar << "\n\n"
         << "Seed: " << seed << "\n" << G4endl;

  G4double thickness = mass / density / 1e2; // layer thickness in [m]
  G4cout << "Calculated cylinder length: " << thickness << " m" << G4endl;

  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine);
  CLHEP::HepRandom::setTheSeed(seed);

  // Construct RunManager and initialize G4 kernel
  G4RunManager *runManager = new G4RunManager();
  runManager->SetUserInitialization(new DetectorConstruction(thickness, density, w_H, w_He, w_N, w_O, w_Ar));
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(particlePDG, energy));
  runManager->Initialize();

  // Get the pointer to the User Interface manager
  G4UImanager* UImanager = G4UImanager::GetUIpointer();
  UImanager->ApplyCommand("/process/inactivate nKiller");
  // UImanager->ApplyCommand("/tracking/verbose 1");

  #ifdef USE_VISUALIZATION
    // Get the pointer to the visualization mmnager
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
