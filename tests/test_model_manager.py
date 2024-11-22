from config import WhisperConfig
from model_manager import WhisperModelManager


class TestModelManager:

    def test_model_manager(self):
        # given
        model_manager = WhisperModelManager(WhisperConfig())

        # when
        model_manager.load_model("large-v3")

        # then
        assert model_manager.loaded_models is not None
