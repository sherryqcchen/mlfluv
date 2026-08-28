import torch
import segmentation_models_pytorch as smp

class SMPSegmentationModel(torch.nn.Module):
    def __init__(
            self,
            architecture="Unet",
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=15,
            num_classes=7,
            num_valid_classes=6,
            encoder_freeze=False,
            temperature=1.0,
            decoder_attention_type="scse",
            encoder_depth=5,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            decoder_aspp_separable=True,
            decoder_aspp_dropout=0.5,
            decoder_segmentation_channels=256,
            upsampling=4,
            aux_params=None,
    ):
        """
        Initialize a segmentation model from segmentation_models_pytorch.

        Args:
            architecture (str): SMP architecture name. Supported here: "Unet", "DeepLabV3Plus", "Segformer".
            encoder_name (str): Name of the encoder backbone (e.g., "resnet34", "mobilenet_v2").
            in_channels (int): Number of input channels (3 for RGB images).
            num_classes (int): Number of output classes (usually 1 for binary segmentation).
            encoder_freeze (bool): Whether to freeze the encoder layers during training.
        """
        super(SMPSegmentationModel, self).__init__()

        architecture_key = architecture.lower()
        if architecture_key == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                decoder_attention_type=decoder_attention_type,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,
                aux_params=aux_params,
            )
        elif architecture_key in {"deeplabv3plus", "deeplabv3+"}:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                encoder_output_stride=encoder_output_stride,
                decoder_channels=decoder_channels,
                decoder_atrous_rates=decoder_atrous_rates,
                decoder_aspp_separable=decoder_aspp_separable,
                decoder_aspp_dropout=decoder_aspp_dropout,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,
                upsampling=upsampling,
                aux_params=aux_params,
            )
        elif architecture_key == "segformer":
            self.model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                decoder_segmentation_channels=decoder_segmentation_channels,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,
                upsampling=upsampling,
                aux_params=aux_params,
            )
        else:
            raise ValueError(f"Unsupported SMP architecture: {architecture}")

        self.num_valid_classes = num_valid_classes
        self.num_classes = num_classes
        self.temperature = temperature
        
        if self.temperature == 0.0:
            print("You cannot do knowledge distillation when temperature is set to 0.")

        # TODO: What is the difference of doing with set encoder_freeze=True in smp.Unet()
        # Freeze encoder
        if encoder_freeze==True:
            self.freeze_encoder()
    
    def load_state_dict(self, state_dict):
        
        # Extract segmentation head's parameters
        segmentation_head_state_dict = dict()
        segmentation_head_state_names =  ('segmentation_head.0.weight', 'segmentation_head.0.bias')
        for k in segmentation_head_state_names:
            segmentation_head_state_dict[k] = state_dict.pop(k)
        
        # Load parameters of the encoder and the decoder 
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        assert set(missing_keys) == set(segmentation_head_state_names)
        assert not unexpected_keys

        # Manually load parameters of the segmentation head
        num_classes = segmentation_head_state_dict["segmentation_head.0.weight"].shape[0]
        # Hack
        with torch.no_grad():
            self.model.segmentation_head[0].weight[:num_classes] = segmentation_head_state_dict["segmentation_head.0.weight"] # new classes weight tensor is filled with random numbers
            self.model.segmentation_head[0].bias[:num_classes] = segmentation_head_state_dict["segmentation_head.0.bias"] # new classes bias is 0
        
        
    def freeze_encoder(self):
        for child in self.model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False
        return
    
    def unfreeze(self):
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = True
        return

    @staticmethod
    def apply_mask(logits, num_valid_classes):
        logits[:, num_valid_classes:] = -torch.inf
        return logits


    def forward(self, *args, **kwargs):
        
        logits = self.model(*args, **kwargs) # [batch_size, n_classes, image_height, image_width]
        logits = self.apply_mask(logits, self.num_valid_classes)
        
        return logits


class SMPUnet(SMPSegmentationModel):
    def __init__(self, encoder_name="resnet34", in_channels=15, num_classes=7, num_valid_classes=6, encoder_freeze=False, temperature=1.0):
        super().__init__(
            architecture="Unet",
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            decoder_attention_type="scse",
            in_channels=in_channels,
            num_classes=num_classes,
            num_valid_classes=num_valid_classes,
            encoder_freeze=encoder_freeze,
            temperature=temperature,
        )


"""
Method to freeze encoder

method 1
model = SMPUnet(..., freeze_encoder=True)

method 2
model = SMPUnet(...)
model.freeze_encoder()

model.unfreeze_encoder()
"""  


# Example usage:
if __name__ == "__main__":
    # Create an instance with encoder freeze
    unet_instance = SMPUnet(encoder_name="resnet34", in_channels=6, num_classes=1, encoder_freeze=True)
    print(unet_instance.model)  # Print the U-Net architecture
