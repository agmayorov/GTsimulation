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

int main(int argc, char *argv[])
{
  // Read input
  // Input example: ./MatterLayer 0 2212 500 60 0.001 H 0 He 0 N 0.75 O 0.25 Ar 0
  if (argc < 8 || argc % 2 != 0) {
    G4cerr << "Wrong number of input parameters" << G4endl;
    return 3;
  }
  // Read values of input variables
  G4long seed = atol(argv[1]);
  G4int particlePDG = atoi(argv[2]); // PDG code of particle
  G4double energy   = atof(argv[3]); // MeV
  G4double mass     = atof(argv[4]); // g/cm^2
  G4double density  = atof(argv[5]); // g/cm^3
  G4int n_element = (argc - 6) / 2; // number of elements
  std::vector<G4String> element_name(n_element);
  std::vector<G4double> element_abundance(n_element);
  for (G4int i = 0; i < n_element; i++) {
    element_name[i] = argv[6 + i * 2];
    element_abundance[i] = atof(argv[7 + i * 2]);
  }
  G4cout << "Input particlePDG: " << particlePDG << "\n"
         << "Input energy: " << energy << " MeV" << "\n"
         << "Input mass: " << mass << " g/cm2" << "\n"
         << "Input density: " << density << " g/cm3" << "\n\n"
         << "Element composition" << G4endl;
  for (G4int i = 0; i < n_element; i++)
    G4cout << element_name[i] << ": " << element_abundance[i] << G4endl;
  G4cout << "\n" << "Seed: " << seed << "\n" << G4endl;

  G4double thickness = mass / density / 1e2; // layer thickness in [m]
  G4cout << "Calculated cylinder length: " << thickness << " m" << G4endl;

  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine);
  CLHEP::HepRandom::setTheSeed(seed);

  // Construct RunManager and initialize G4 kernel
  G4RunManager *runManager = new G4RunManager();
  runManager->SetUserInitialization(new DetectorConstruction(thickness, density, element_name, element_abundance));
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(particlePDG, energy));
  runManager->Initialize();

  // Get the pointer to the User Interface manager
  G4UImanager *UImanager = G4UImanager::GetUIpointer();
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
