import argparse

import torch.utils.data

from Data.dataset import ich_dataset
from Nets.Api import run_net
from Nets.Joint_Attention_Fusion import Cross_Modal_Attention_Fusion, Multi_Head_Self_Attention_Fusion,Text_Representation_Transformation,Vision_treatment_Net
from Nets.Vision_Encoder import get_pretrained_Vision_Encoder,get_Vision_Encoder
from Nets.Text_Encoder import get_Text_Encoder
from Nets.Classification_Header import get_Classification_Header,get_pretrained_Classification_Header



def prepare_to_train_net(batch_size, epochs, learning_rate, use_gpu, gpu_id, vision_pre, classification_pre,
                         loss_function_id, fusion_structure_id, parameters=[512, 256, 256, 256]):
    """
    Prepares and launches the training process for the neural network.

    Parameters:
    - batch_size (int): The size of the batches used in training.
    - epochs (int): The number of epochs for training.
    - learning_rate (float): The learning rate for the optimizer.
    - use_gpu (bool): Flag to indicate whether to use GPU.
    - gpu_id (str): The ID of the GPU to use.
    - vision_pre (bool): Whether to use a pre-trained vision encoder.
    - classification_pre (bool): Whether to use a pre-trained classification header.
    - loss_function_id (int): ID of the loss function to use.
    - fusion_structure_id (int): ID of the fusion structure to use.
    - parameters (list): List of parameters for network components.
    """
    # Clear GPU memory cache
    torch.cuda.empty_cache()
    
    # Initialize the text encoder
    text_encoder = get_Text_Encoder()
    
    # Initialize the vision encoder, with an option to use pre-trained weights
    if vision_pre:
        vision_encoder = get_pretrained_Vision_Encoder()
    else:
        vision_encoder = get_Vision_Encoder()
    
    # Initialize the transformation and attention components
    text_trt = Text_Representation_Transformation(64, 128, 32, 32, 3)
    vision_trt = Vision_treatment_Net()
    cross_atten = Cross_Modal_Attention_Fusion(parameters[0], parameters[1], 1, 256, [256, 256, 256])
    self_atten = Multi_Head_Self_Attention_Fusion(16, parameters[3], parameters[4], 0)
    
    # Initialize the classification header, with an option to use pre-trained weights
    if classification_pre:
        classify_head = get_pretrained_Classification_Header()
    else:
        classify_head = get_Classification_Header()
    
    # Collect all parameters that require gradients for the optimizer
    params = (
        [p for p in text_encoder.parameters() if p.requires_grad] +
        [p for p in vision_encoder.parameters() if p.requires_grad] +
        [p for p in text_trt.parameters() if p.requires_grad] +
        [p for p in vision_trt.parameters() if p.requires_grad] +
        [p for p in cross_atten.parameters() if p.requires_grad] +
        [p for p in self_atten.parameters() if p.requires_grad] +
        [p for p in classify_head.parameters() if p.requires_grad]
    )

    # Print the number of parameters in each component for debugging
    text_encoder_params = [p for p in text_encoder.parameters() if p.requires_grad]
    print("Text Encoder parameters: " + str(len(text_encoder_params)))
    vision_encoder_params = [p for p in vision_encoder.parameters() if p.requires_grad]
    print("Vision Encoder parameters: " + str(len(vision_encoder_params)))

    text_trt_params = [p for p in text_trt.parameters() if p.requires_grad]
    print("Text TRT parameters: " + str(len(text_trt_params)))
    vision_trt_params = [p for p in vision_trt.parameters() if p.requires_grad]
    print("Vision TRT parameters: " + str(len(vision_trt_params)))
    cross_atten_params = [p for p in cross_atten.parameters() if p.requires_grad]
    print("Cross Attention parameters: " + str(len(cross_atten_params)))
    self_atten_params = [p for p in self_atten.parameters() if p.requires_grad]
    print("Self Attention parameters: " + str(len(self_atten_params)))

    classify_params = [p for p in classify_head.parameters() if p.requires_grad]
    print("Classification Header parameters: " + str(len(classify_params)))

    # Load the dataset
    data_set = ich_dataset("Data/all.txt")
    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    print("Datasets loaded, dataset size: %d" % len(data_set))

    # Set up the optimizer
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Set up the device (CPU or GPU)
    if use_gpu:
        device = torch.device(('cuda:' + gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print("Using device: {}".format(device))
    print("Preparation completed! Launching training! 🚀")
    print("Using full Nets to classify")

    # Start the training process
    run_net(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        text_trt=text_trt,
        vision_trt=vision_trt,
        self_attn=self_atten,
        cross_attn=cross_atten,
        classify=classify_head,
        device=device,
        epochs=epochs,
        data_loader=data_loader,
        optimizer=optimizer,
        loss_function_id=loss_function_id,
        fusion_structure_id=fusion_structure_id,
        batch_size=batch_size
    )


if __name__ == "__main__":
    # Adding necessary input arguments for the script
    parser = argparse.ArgumentParser(description="Add arguments for testing")
    parser.add_argument("--batch_size", default=4, help="Batch size for training", type=int)
    parser.add_argument("--epochs", default=300, help="Number of epochs for training", type=int)
    parser.add_argument("--use_gpu", default=True, help="Device choice, if CUDA isn't available, program will warn", type=bool)
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--gpu_id", default="0", help="GPU ID to use", type=str)
    parser.add_argument("--classification_pre", help="Use pre-trained classification header", action='store_true')
    parser.add_argument("--vision_pre", help="Use pre-trained vision encoder", action='store_true')
    parser.add_argument("--loss_function_id", help="ID of the loss function to use", default=0, type=int)
    parser.add_argument("--fusion_structure_id", help="ID of the fusion structure to use", default=0, type=int)
    args = parser.parse_args()

    print(args)

    # Prepare and launch the training process with the provided arguments
    prepare_to_train_net(
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_gpu=args.use_gpu,
        learning_rate=args.learning_rate,
        gpu_id=args.gpu_id,
        vision_pre=args.vision_pre,
        classification_pre=args.classification_pre,
        loss_function_id=args.loss_function_id,
        fusion_structure_id=args.fusion_structure_id
    )
