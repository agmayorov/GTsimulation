#include "G4RunManager.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

#include "G4UImanager.hh"
#include <chrono>

using namespace Atmosphere;

int main(int argc, char* argv[])
{
  // Read input
  // Input example: ./Atmosphere 2212 10000 95 15 1 0 0 0 150 150 4
  if (argc != 12) {
    G4cout << "Wrong number of input parameters" << G4endl;
    return 0;
  }
  // Read values of input variables
  G4int particlePDG = atoi(argv[1]); // PDG code of particle
  G4double energy   = atof(argv[2]); // MeV
  G4double height   = atof(argv[3]); // km
  G4double alpha    = atof(argv[4]); // degrees
  G4int doy         = atoi(argv[5]); // day of year
  G4double sec      = atof(argv[6]); // sec
  G4double lat      = atof(argv[7]); // degrees
  G4double lon      = atof(argv[8]); // degrees
  G4double f107A    = atof(argv[9]);
  G4double f107     = atof(argv[10]);
  G4double ap       = atof(argv[11]);
  G4cout << "Input particlePDG: " << particlePDG << '\n'
         << "Input energy: " << energy << " GeV" << '\n'
         << "Input height: " << height << " km" << '\n'
         << "Input alpha: " << alpha << " degrees" << '\n'
         << "Input day of year: " << doy << '\n'
         << "Input number of seconds in day: " << sec << " sec" << '\n'
         << "Input geodetic latitude: " << lat << " degrees" << '\n'
         << "Input geodetic longitude: " << lon << " degrees" << '\n'
         << "Input average of F10.7 flux: " << f107A << '\n'
         << "Input daily F10.7 flux: " << f107 << '\n'
         << "Input ap magnetic index: " << ap << G4endl;

  // Preliminary calculations
  alpha = alpha/180.*CLHEP::pi;
  G4double Rm = 200.; // km
  G4double R = Rm/cos(alpha) + height*tan(alpha);

  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine);
  auto seed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  CLHEP::HepRandom::setTheSeed(seed);

  // Construct RunManager and initialize G4 kernel
  G4RunManager *runManager = new G4RunManager();
  runManager->SetUserInitialization(new DetectorConstruction(R, doy, sec, lat, lon, f107A, f107, ap));
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(particlePDG, energy, height, alpha));
  runManager->Initialize();

  // Get the pointer to the User Interface manager
  G4UImanager* UImanager = G4UImanager::GetUIpointer();
  UImanager->ApplyCommand("/process/inactivate nKiller");
  // UImanager->ApplyCommand("/tracking/verbose 1");

  // Run 1 particle
  runManager->BeamOn(1);

  // Job termination
  delete runManager;

  return 0;
}
