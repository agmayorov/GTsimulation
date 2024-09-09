#include "DetectorConstruction.hh"

namespace MatterLayer
{

DetectorConstruction::DetectorConstruction(G4double thickness, G4double density, \
  G4double w_H, G4double w_He, G4double w_N, G4double w_O, G4double w_Ar)
: G4VUserDetectorConstruction(),
  fThickness(thickness),
  fDensity(density),
  fW_H(w_H),
  fW_He(w_He),
  fW_N(w_N),
  fW_O(w_O),
  fW_Ar(w_Ar)
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  G4bool checkOverlaps = true;

  G4double z = 0.; // atomic number
  G4double a = 0.; // atomic mass
  G4double density = 0.;
  G4int nelem = 0;

  G4Element* H  = new G4Element("Hydrogen", "H",  z =  1., a = 1.008*g/mole);
  G4Element* He = new G4Element("Helium",   "He", z =  2., a = 4.003*g/mole);
  G4Element* N  = new G4Element("Nitrogen", "N",  z =  7., a = 14.01*g/mole);
  G4Element* O  = new G4Element("Oxygen",   "O",  z =  8., a = 16.00*g/mole);
  G4Element* Ar = new G4Element("Argon",    "Ar", z = 18., a = 39.95*g/mole);

  //--------------------------------------- World --------------------------------------------------

  G4Material* world_mat = new G4Material("Air", density = fDensity*g/cm3, nelem = 5);
  world_mat->AddElement(H,  fW_H *100.*perCent);
  world_mat->AddElement(He, fW_He*100.*perCent);
  world_mat->AddElement(N,  fW_O *100.*perCent);
  world_mat->AddElement(O,  fW_N *100.*perCent);
  world_mat->AddElement(Ar, fW_Ar*100.*perCent);

  G4double world_sizeR =    fThickness*m;
  G4double world_sizeZ = 2.*fThickness*m;

  G4Tubs* solidWorld =
    new G4Tubs("World",
               0.*m,                      // inner radius
               world_sizeR,               // outer radius
               0.5*world_sizeZ,           // hight
               0.*deg,                    // start angle
               360.*deg);                 // spanning angle

  G4LogicalVolume* logicWorld = 
    new G4LogicalVolume(solidWorld,       // its solid
                        world_mat,        // its material
                        "World");         // its name

  G4VPhysicalVolume* physWorld = 
    new G4PVPlacement(0,                  // no rotation
                      G4ThreeVector(),    // at (0,0,0)
                      logicWorld,         // its logical volume
                      "World",            // its name
                      0,                  // its mother volume
                      false,              // no boolean operation
                      0,                  // copy number
                      checkOverlaps);     // overlaps checking

  return physWorld;
}

}
