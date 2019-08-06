from common import model

my_model = model.MyModel(
  arch='Unet',
  checkpoint='unet'
)

my_model.get_model_summary()
