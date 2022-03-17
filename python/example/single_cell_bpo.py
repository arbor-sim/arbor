# Read BluePyOpt cell output in mixed .json/.acc form created with `create_acc` into Arbor objects

import arbor
import os, json


def read_cell(cell_json_filename):

    with open(cell_json_filename) as cell_json_file:
        cell_json = json.load(cell_json_file)

    cell_json_dir = os.path.dirname(cell_json_filename)

    morphology_filename = os.path.join(cell_json_dir, cell_json['morphology'])
    if morphology_filename.endswith('.swc'):
        morpho = arbor.load_swc_arbor(morphology_filename)
    elif morphology_filename.endswith('.asc'):
        morpho = arbor.load_asc(morphology_filename)
    else:
        raise RuntimeError('Unsupported morphology {} (only .swc and .asc supported)'.format(morphology_filename))
    
    labels = arbor.load_component(os.path.join(cell_json_dir, cell_json['label_dict'])).component
    decor = arbor.load_component(os.path.join(cell_json_dir, cell_json['decor'])).component

    return cell_json, morpho, labels, decor