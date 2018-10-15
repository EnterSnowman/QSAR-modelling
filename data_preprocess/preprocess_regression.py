import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdDistGeom
from rdkit.ML.Descriptors import MoleculeDescriptors


def compute_and_save_all_descriptors(path_to_data):
    datasets = ['train', 'test']
    nms = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
    for d in datasets:
        columns = ['SMILES']
        columns.extend(nms)
        final_df = pd.DataFrame(columns=columns)
        df = pd.read_csv(path_to_data + 't_pyriformis_' + d + '.csv')[['SMILES', 'log(IGC50-1) {measured, converted}']]
        print(d + " process")
        for index, row in df['SMILES'].iteritems():
            mol = Chem.MolFromSmiles(row)
            descs = calc.CalcDescriptors(mol)
            v = [row]
            v.extend(list(descs))
            # get 3D descriptors
            rdDescs, ds = calculate_rdMolDescriptors(mol)
            v.extend(rdDescs)
            cols = []
            cols.extend(columns)
            cols.extend(ds)
            new_row = pd.DataFrame(np.array([v]), columns=cols)
            final_df = final_df.append(new_row, ignore_index=True)
        final_df['log(IGC50-1) {measured, converted}'] = df['log(IGC50-1) {measured, converted}']
        final_df.dropna(inplace=True)
        print(final_df)
        final_df.to_csv(path_to_data + 't_pyriformis_' + d + '_all_descriptors.csv')


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


compute_and_save_all_descriptors('../data/regression/')
