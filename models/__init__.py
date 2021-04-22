# When called as "from models import ____", this is what becomes available
# Please register here new models as they are implemented!!
# Also update the get_model_instance function
from .simple_sis import SimpleSIS
from .double_sis import DoubleSIS


def get_model_instance(model_class, input_dict):

    params = list()

    if model_class is SimpleSIS:
        params.append(float(input_dict["beta"]))  # Infection probability
        params.append(float(input_dict["mu"]))  # Healing probability

    elif model_class is DoubleSIS:
        # Relative time scale parameter pi, as optional
        if "pi" in input_dict:
            pi = float(input_dict["pi"])
            ompi = 1.0 - pi
        else:
            pi = ompi = 1.0

        params.append(ompi * float(input_dict["beta1"]))  # Infection probability - disease 1
        params.append(pi * float(input_dict["beta2"]))  # Infection probability - disease 2
        params.append(ompi * float(input_dict["mu1"]))  # Healing probability - disease 1
        params.append(pi * float(input_dict["mu2"]))  # Healing probability - disease 2
        params.append(float(input_dict["gamma1"]))  # Susceptibility change to disease 1 (by having 2)
        params.append(float(input_dict["gamma2"]))  # Susceptibility change to disease 2 (by having 1)

    else:
        raise ValueError("Hey, model class '{}' must be registered in function 'get_model_instance()' of the "
                         "__init__.py file of the 'models' directory.")

    return model_class(*params)

