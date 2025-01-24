#include "G4RunManager.hh"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

#include "G4UImanager.hh"
#include <chrono>

#ifdef USE_VISUALIZATION
  #include "G4UIExecutive.hh"
  #include "G4VisExecutive.hh"
#endif

using namespace Atmosphere;

int main(int argc, char* argv[])
{
  // Read input
  // Input example: ./Atmosphere 2212 10000 0 0 6480 0 0 -1 6400 1 0 0 0 150 150 4
  if (argc != 17) {
    G4cout << "Wrong number of input parameters" << G4endl;
    return 0;
  }
  // Read values of input variables
  G4int particlePDG    = atoi(argv[1]); // PDG code of particle
  G4double energy      = atof(argv[2]); // MeV
  G4double X           = atof(argv[3]); // km
  G4double Y           = atof(argv[4]); // km
  G4double Z           = atof(argv[5]); // km
  G4double Vx          = atof(argv[6]);
  G4double Vy          = atof(argv[7]);
  G4double Vz          = atof(argv[8]);
  G4double earthRadius = atof(argv[9]); // km
  G4int doy            = atoi(argv[10]); // day of year
  G4double sec         = atof(argv[11]); // sec
  G4double lat         = atof(argv[12]); // degrees
  G4double lon         = atof(argv[13]); // degrees
  G4double f107A       = atof(argv[14]);
  G4double f107        = atof(argv[15]);
  G4double ap          = atof(argv[16]);
  G4cout << "Input particlePDG: " << particlePDG << '\n'
         << "Input energy: " << energy << " GeV" << '\n'
         << "Input X, Y, Z: " << X << ' ' << Y << ' '  << Z << " km" << '\n'
         << "Input Vx, Vy, Vz: " << Vx << ' ' << Vy << ' '  << Vz << '\n'
         << "Input Earth radius: " << earthRadius << " km" << '\n'
         << "Input day of year: " << doy << '\n'
         << "Input number of seconds in day: " << sec << " sec" << '\n'
         << "Input geodetic latitude: " << lat << " degrees" << '\n'
         << "Input geodetic longitude: " << lon << " degrees" << '\n'
         << "Input average of F10.7 flux: " << f107A << '\n'
         << "Input daily F10.7 flux: " << f107 << '\n'
         << "Input ap magnetic index: " << ap << G4endl;

  // Preliminary definitions and calculations
  G4ThreeVector coordinates(X*km, Y*km, Z*km);
  G4ThreeVector velocity(Vx, Vy, Vz);
  G4double lst = sec / 3600. + lon / 15.;
  nrlmsise_input input = {
    0,        /* year, currently ignored */
    doy,      /* day of year */
    sec,      /* seconds in day (UT) */
    0,        /* altitude in kilometers */
    lat,      /* geodetic latitude */
    lon,      /* geodetic longitude */
    lst,      /* local apparent solar time (hours) */
    f107A,    /* 81 day average of F10.7 flux (centered on doy) */
    f107,     /* daily F10.7 flux for previous day */
    ap,       /* magnetic index(daily) */
    nullptr,  /* ap_array */
  };

  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine);
  auto seed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  CLHEP::HepRandom::setTheSeed(seed);

  // Construct RunManager and initialize G4 kernel
  G4RunManager *runManager = new G4RunManager();
  runManager->SetUserInitialization(new DetectorConstruction(earthRadius, input));
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(particlePDG, energy, coordinates, velocity));
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
