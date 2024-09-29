#include "G4RunManager.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

#include <chrono>

using namespace DecayGenerator;

int main(int argc, char* argv[])
{
  // Read input
  // Input example: ./DecayGenerator 15 500
  if (argc != 3) {
    G4cout << "Wrong number of input parameters" << G4endl;
    return 0;
  }
  // Read values of input variables
  G4int particlePDG = atoi(argv[1]); // PDG code of particle
  G4double energy   = atof(argv[2]); // MeV
  G4cout << "Input particlePDG: " << particlePDG << "\n"
         << "Input energy: " << energy << " MeV" << G4endl;

  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine);
  auto seed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  CLHEP::HepRandom::setTheSeed(seed);

  // Construct RunManager and initialize G4 kernel
  G4RunManager *runManager = new G4RunManager();
  runManager->SetUserInitialization(new DetectorConstruction());
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(particlePDG, energy));
  runManager->Initialize();

  // Run 1 particle
  runManager->BeamOn(1);

  // Job termination
  delete runManager;

  return 0;
}
