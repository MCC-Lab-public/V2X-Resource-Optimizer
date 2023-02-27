import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

DDPG_AUTO_CONFIGSPACE = {
    # NAS
    "hidden_layer_a1": CSH.UniformIntegerHyperparameter("hidden_layer_a1", lower=1, upper=1000, default_value=400, log=True),
    "hidden_layer_a2": CSH.UniformIntegerHyperparameter("hidden_layer_a2", lower=1, upper=1000, default_value=300, log=True),
    "hidden_layer_c1": CSH.UniformIntegerHyperparameter("hidden_layer_c1", lower=1, upper=1000, default_value=400, log=True),
    "hidden_layer_c2": CSH.UniformIntegerHyperparameter("hidden_layer_c2", lower=1, upper=1000, default_value=400, log=True),
    # HPO
    "lr_a": CSH.UniformFloatHyperparameter('lr_a', lower=1e-5, upper=1e-1, default_value=1e-3, log=True),
    "lr_c": CSH.UniformFloatHyperparameter('lr_c', lower=1e-5, upper=1e-1, default_value=1e-3, log=True),
    "tau": CSH.UniformFloatHyperparameter('tau', lower=0.0, upper=1.0, default_value=0.001, log=True),
    "gamma": CSH.UniformFloatHyperparameter('gamma', lower=0.0, upper=1.0, default_value=0.99, log=True),
    "target_update_a": CSH.UniformIntegerHyperparameter("target_update_a", lower=1, upper=20, default_value=10, log=True),
    "target_update_c": CSH.UniformIntegerHyperparameter("target_update_c", lower=1, upper=20, default_value=10, log=True)
}