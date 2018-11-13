from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np


def compute_features(smiles, features):
    print(features)
    nms = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
    mol = Chem.MolFromSmiles(smiles)
    descs = calc.CalcDescriptors(mol)
    v = []
    v.extend(list(descs))
    rdDescs, ds = calculate_rdMolDescriptors(mol)
    v.extend(rdDescs)
    cols = []
    cols.extend(nms)
    cols.extend(ds)
    d = dict(zip(cols, v))
    res = [d[f] for f in features]
    print(res)
    return np.array(res)


def calculate_rdMolDescriptors(mol):
    rdDescs = []
    d = []

    # conformers

    # m2 = Chem.AddHs(mol)
    # # use the original distance geometry + minimisation method
    # print(AllChem.EmbedMolecule(m2))
    # AllChem.UFFOptimizeMolecule(m2)
    # m3 = Chem.AddHs(mol)
    # # use the new method
    # print(AllChem.EmbedMolecule(m3, AllChem.ETKDG()))
    list_desc_len = [80, 210, 224, 114, 273]
    m3 = Chem.AddHs(mol)
    id = AllChem.EmbedMolecule(m3, useRandomCoords=True, maxAttempts=1000)
    # print(id)
    d.append('PBF')
    d.append('PMI1')
    d.append('PMI2')
    d.append('PMI3')
    d.append('NPR1')
    d.append('NPR2')
    d.append('RadiusOfGyration')
    d.append('InertialShapeFactor')
    d.append('Eccentricity')
    d.append('Asphericity')
    d.append('SpherocityIndex')
    d.extend(['AUTOCORR3D_' + str(i) for i in range(0, list_desc_len[0])])
    d.extend(['RDF_' + str(i) for i in range(0, list_desc_len[1])])
    d.extend(['MORSE_' + str(i) for i in range(0, list_desc_len[2])])
    d.extend(['WHIM_' + str(i) for i in range(0, list_desc_len[3])])
    d.extend(['GETAWAY_' + str(i) for i in range(0, list_desc_len[4])])
    if id >= 0:
        AllChem.MMFFOptimizeMolecule(m3)
        rdDescs.append(rdMolDescriptors.CalcPBF(m3))
        rdDescs.append(rdMolDescriptors.CalcPMI1(m3))
        rdDescs.append(rdMolDescriptors.CalcPMI2(m3))
        rdDescs.append(rdMolDescriptors.CalcPMI3(m3))
        rdDescs.append(rdMolDescriptors.CalcNPR1(m3))
        rdDescs.append(rdMolDescriptors.CalcNPR2(m3))
        rdDescs.append(rdMolDescriptors.CalcRadiusOfGyration(m3))
        rdDescs.append(rdMolDescriptors.CalcInertialShapeFactor(m3))
        rdDescs.append(rdMolDescriptors.CalcEccentricity(m3))
        rdDescs.append(rdMolDescriptors.CalcAsphericity(m3))
        rdDescs.append(rdMolDescriptors.CalcSpherocityIndex(m3))

        # lists
        rdDescs.extend(rdMolDescriptors.CalcAUTOCORR3D(m3))
        rdDescs.extend(rdMolDescriptors.CalcRDF(m3))
        rdDescs.extend(rdMolDescriptors.CalcMORSE(m3))
        rdDescs.extend(rdMolDescriptors.CalcWHIM(m3))
        rdDescs.extend(rdMolDescriptors.CalcGETAWAY(m3))
    else:
        for i in range(0, 11 + sum(list_desc_len)):
            rdDescs.append(np.NaN)

    return rdDescs, d
