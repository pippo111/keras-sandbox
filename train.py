from common import model

my_model = model.MyModel(
  arch='Unet3d',
  checkpoint='unet_3d'
)

my_model.get_model_summary()
