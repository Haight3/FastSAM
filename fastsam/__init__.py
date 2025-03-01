# flake8: noqa
# Ultralytics YOLO 🚀, AGPL-3.0 license

# from .val import FastSAMValidator
import warnings

import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from mlops.trainer import BasePythonModel, MLFlowLightningModule

from .decoder import FastSAMDecoder
from .model import FastSAM
from .predict import FastSAMPredictor
from .prompt import FastSAMPrompt

__all__ = (
    "FastSAMPredictor",
    "FastSAM",
    "FastSAMPrompt",
    "FastSAMDecoder",
    "FastSAMModel",
)


class FastSAMWrapper(BasePythonModel):
    def load_context(self, context):
        # Initialize the model with the stored parameters
        self.state.pop("ckpt_path", None)

        self.model = self.model_class(
            ckpt_path=context.artifacts["checkpoint"],
        )

        self.model.eval()

    def predict(self, context, model_input) -> np.ndarray:

        # Input ============================================================================
        # Extract patch, height, and width from the input dictionary
        patch = model_input["patch"]  # Input Patch

        # Make Prediction ==================================================================
        # Make predictions with the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            output = self.model.predict(patch)

        return output

    def get_signature(self, **kwargs):

        # Define input schema (considering patch as an object and height, width as integers)
        input_schema = Schema(
            [
                TensorSpec(
                    np.dtype(np.uint8), (-1, -1, 3), name="patch"
                ),  # Dynamic H and W
            ]
        )

        # Define output schema (assuming the output is a tensor or numerical value)
        output_schema = Schema(
            [TensorSpec(np.dtype(np.float32), (-1, -1, 3), name="output")]
        )  # You can adjust this based on your model output

        # Create the signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        return signature


class FastSAMModel(MLFlowLightningModule):
    def __init__(
        self,
        ckpt_path=None,
        retina_masks: bool = True,
        imgsz: int = 512,
        conf: float = 0.15,
        iou: float = 0.95,
    ):

        super().__init__()
        # Utility ==========================================================================
        self.set_state(**locals())
        self.name = "FastSamX"
        self.wrapper = FastSAMWrapper

        # Build Model ======================================================================
        self.model = None
        if ckpt_path is not None:
            self.model = FastSAM(ckpt_path)

        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.ckpt_path = ckpt_path

    def predict(self, x: np.array):

        everything_results = self.model(
            x,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        prompt_process = FastSAMPrompt(x, everything_results, device=self.device)
        ann = prompt_process.everything_prompt()

        if len(ann) > 0:
            result = prompt_process.to_mask(
                annotations=ann,
                mask_random_color=True,
                better_quality=True,
                retina=True,
            )

        else:
            result = np.zeros(x.shape)

        return result
