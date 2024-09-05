#include "DetectorConstruction.hh"

namespace DecayGenerator
{

DetectorConstruction::DetectorConstruction()
: G4VUserDetectorConstruction()
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  //--------------------------------------- World --------------------------------------------------

  G4Sphere *world_svol = new G4Sphere("world", 0., 1.*mm, 0., 2 * CLHEP::pi, 0., CLHEP::pi);

  G4Material *world_mat = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4LogicalVolume *world_lvol = new G4LogicalVolume(world_svol, world_mat, "world");

  G4VPhysicalVolume *world_pvol = new G4PVPlacement(nullptr, G4ThreeVector(), world_lvol, "world", nullptr, false, 0, false);

  return world_pvol;
}

}
