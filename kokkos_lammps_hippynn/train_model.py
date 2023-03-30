import torch
import ase.io
import numpy as np
torch.set_default_dtype(torch.float32)

import hippynn
hippynn.settings.WARN_LOW_DISTANCES = False

max_epochs=2000

network_params = {
    "possible_species": [0, 47],
    "n_features": 8,
    "n_sensitivities": 16,
    "dist_soft_min": 2.3,
    "dist_soft_max": 3.75,
    "dist_hard_max": 4,
    "n_interaction_layers": 1,
    "n_atom_layers": 3,
    "sensitivity_type": "inverse",
    "resnet": True,
}
early_stopping_key = "Loss-Err"
test_size = 0.1
valid_size = 0.1

def get_dataset(db_info):
    # Dataset processing
    try:
        atom_list=ase.io.read('./Ag_warm_nospin.xyz',index=":",format='extxyz')
    except FileNotFoundError:
        raise FileNotFoundError("Required file Ag_warm_nospin.xyz not found. See README.")
    def get_property(atom_list,f):
        return np.asarray([f(a) for a in atom_list])

    props={}
    props["Z"]=get_property(atom_list, lambda x:x.numbers)
    props["F"]=get_property(atom_list, lambda x:x.get_forces())
    props["E"]=get_property(atom_list, lambda x:x.get_potential_energy())
    props["R"]=get_property(atom_list, lambda x:x.get_positions())
    props["Cell"]=get_property(atom_list, lambda x:x.get_cell())
    n_atoms = (props["Z"]!=0).sum(axis=1)
    props["EpA"]=props["E"]/n_atoms

    for k,v in props.items():
        print(k,v.shape,v.dtype)
        if v.dtype==np.float64:
            props[k]=v.astype(np.float32)
    ### End Dataset Processing
    
    from hippynn.databases import Database

    database = Database(
        arr_dict=props,
        seed=1001,  # Random seed for splitting data
        quiet=False,
        pin_memory=False,
        test_size=test_size,
        valid_size=valid_size,
        **db_info
    )
    database.send_to_device() # Send to GPU

    return database,atom_list[0]

def setup_network(network_params):

    # Hyperparameters for the network
    print("Network hyperparameters:")
    print(network_params)

    from hippynn.graphs import inputs, networks, targets, physics

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    cell = inputs.CellNode(db_name="Cell")

    network = networks.HipnnQuad("HIPNN", (species, positions, cell), module_kwargs=network_params, periodic=True)

    henergy = targets.HEnergyNode("HEnergy", network)
    sys_energy = henergy.mol_energy
    sys_energy.db_name = "E"
    hierarchicality = henergy.hierarchicality
    hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
    force = physics.GradientNode("forces", (sys_energy, positions), sign=-1)
    force.db_name = "F"

    en_peratom = physics.PerAtom("T/Atom", sys_energy)
    en_peratom.db_name = "EpA"

    from hippynn.graphs import loss

    rmse_energy = loss.MSELoss(en_peratom.pred, en_peratom.true) ** (1 / 2)
    mae_energy = loss.MAELoss(en_peratom.pred, en_peratom.true)
    rsq_energy = loss.Rsq(en_peratom.pred, en_peratom.true)
    force_rsq = loss.Rsq(force.pred, force.true)
    force_mse = loss.MSELoss(force.pred, force.true)
    force_mae = loss.MAELoss(force.pred, force.true)
    force_rmse = force_mse ** (1 / 2)
    rbar = loss.Mean(hierarchicality.pred)
    l2_reg = loss.l2reg(network)

    loss_error = 10 * (rmse_energy + mae_energy) + (force_mae + force_rmse)
    loss_regularization = 1e-6 * l2_reg + rbar
    train_loss = loss_error + loss_regularization

    # Factors of 1e3 for meV
    validation_losses = {
        "EpA-RMSE": 1e3*rmse_energy,
        "EpA-MAE": 1e3*mae_energy,
        "EpA-RSQ": rsq_energy,
        "ForceRMSE": 1e3*force_rmse,
        "ForceMAE": 1e3*force_mae,
        "ForceRsq": force_rsq,
        "T-Hier": rbar,
        "L2Reg": l2_reg,
        "Loss-Err": loss_error,
        "Loss-Reg": loss_regularization,
        "Loss": train_loss,
    }

    from hippynn.experiment.assembly import assemble_for_training
    training_modules, db_info = assemble_for_training(train_loss, validation_losses)
    
    return henergy, training_modules, db_info

def fit_model(training_modules,database):

    model, loss_module, model_evaluator = training_modules

    from hippynn.pretraining import set_e0_values
    set_e0_values(henergy, database, peratom=True, energy_name="EpA", decay_factor=1e-2)

    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

    scheduler = RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=128,
        patience=25,
        factor=0.5,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=4,
        eval_batch_size=128,
        max_epochs=max_epochs,
        termination_patience=50,
        stopping_key=early_stopping_key,
    )

    from hippynn.experiment import SetupParams, setup_and_train, test_model

    experiment_params = SetupParams(controller=controller)

    print("Experiment Params:")
    print(experiment_params)

    # Parameters describing the training procedure.

    setup_and_train(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
    )

    with hippynn.tools.log_terminal("model_results.txt",'wt'):
        test_model(database, training_modules.evaluator, 128, "Final Training")
    

if __name__=="__main__":
    print("Setting up model.")
    henergy, training_modules, db_info = setup_network(network_params)
    
    print("Preparing dataset.")
    database, first_frame = get_dataset(db_info)

    print("Training model")
    fit_model(training_modules,database)

    print("Exporting lammps interfaice")
    ase.io.write('ag_box.data',first_frame,format='lammps-data')
    from hippynn.interfaces.lammps_interface import MLIAPInterface
    unified = MLIAPInterface(henergy, ["Ag"], model_device=torch.cuda.current_device())
    torch.save(unified, "hippynn_lammps_model.pt")    
    print("All done.")
