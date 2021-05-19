#import plaidml.keras
#plaidml.keras.install_backend()

from experiments import experiments

model_name = ["GAIN", "VAE", "GAIN_embedding", "VAE_embedding"]

for i in model_name:
    model_experiment = experiments(model_name = i)
    model_experiment.run_model()
