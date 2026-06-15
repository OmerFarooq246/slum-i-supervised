import keras
from EfficientNet_Segmenters.Decoder import Decoder

class Encoder(keras.layers.Layer):
	def __init__(self, input_shape, **kwargs):
		super().__init__(**kwargs)
		base_model = keras.applications.EfficientNetB0(
			include_top=False,
			weights="imagenet",
			input_shape=input_shape,
		)
		self.encoder_layers = [
            base_model.layers[17],
            base_model.layers[46],
            base_model.layers[75],
            base_model.layers[119],
            base_model.layers[162],
            base_model.layers[221],
            base_model.layers[234],
        ]
		encoder_outputs = [layer.output for layer in self.encoder_layers]
		self.encoder_model = keras.Model(base_model.input, encoder_outputs)
		del base_model

	def call(self, pixel_values):
		encoder_outputs = self.encoder_model(pixel_values)
		return encoder_outputs
	

class EfficientNetB0_Segmenter(keras.Model):
	def __init__(self, config, **kwargs):
		super().__init__(**kwargs)
		self.encoder = Encoder(config.input_shape)
		self.decoder = Decoder(
			config.decoder_dims,
			config.encoder_blocks,
			config.dropout_ratio,
			config.num_classes,
			config.output_size,
			config.decoder_upscale_size #for B5 = [256, 256]
		)
	def call(self, pixel_values):
		encoder_outputs = self.encoder(pixel_values)
		decoder_output = self.decoder(encoder_outputs)
		return decoder_output