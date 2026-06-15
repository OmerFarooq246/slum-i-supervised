import keras
from EfficientNet_Segmenters.Decoder import Decoder

class EncoderB6(keras.layers.Layer):
	def __init__(self, input_shape, **kwargs):
		super().__init__(**kwargs)
		base_model = keras.applications.EfficientNetB6(
			include_top=False,
			weights="imagenet",
			input_shape=input_shape,
		)
		self.encoder_layers = [
            base_model.layers[41],
            base_model.layers[130],
            base_model.layers[219],
            base_model.layers[338],
            base_model.layers[456],
            base_model.layers[620],
            base_model.layers[663],
        ]
		encoder_outputs = [layer.output for layer in self.encoder_layers]
		self.encoder_model = keras.Model(base_model.input, encoder_outputs)
		del base_model

	def call(self, pixel_values):
		encoder_outputs = self.encoder_model(pixel_values)
		return encoder_outputs
	

class EfficientNetB6_Segmenter(keras.Model):
	def __init__(self, config, **kwargs):
		super().__init__(**kwargs)
		self.encoder = EncoderB6(config.input_shape)
		self.decoder = Decoder(
			config.decoder_dims,
			config.encoder_blocks,
			config.dropout_ratio,
			config.num_classes,
			config.output_size,
			config.decoder_upscale_size #for B6 = [256, 256]
		)
	def call(self, pixel_values):
		encoder_outputs = self.encoder(pixel_values)
		decoder_output = self.decoder(encoder_outputs)
		return decoder_output