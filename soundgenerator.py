from preprocessing import MinMaxNormalizer
import librosa

class SoundGenerator:
    """generate audio from spectrograms"""

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormalizer(0, 1)

    def generate(self, spectrograms, min_max_values):
        """spectrograms is a list of arrays"""
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        # zip to iterate the tuple
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape to 2 dim
            log_spectrogram = spectrogram[:, :, 0]
            # denormalize
            denorm_log_spec = self._min_max_normalizer.denormalize(log_spectrogram, min_max_value["min"], min_max_value["max"])
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # apply ISTFT or Griffin-Lim
            #signal = librosa.istft(spec, hop_length=self.hop_length)
            signal = librosa.griffinlim(spec, hop_length=self.hop_length)
            signals.append(signal)
        return signals