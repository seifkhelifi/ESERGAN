from flask import Flask, request, jsonify, Response


from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from model import Generator


    
device = "cpu" 
high_res = 1024
low_res = high_res // 4
num_channels = 3


app = Flask(__name__)


@app.route("/generate_high_res", methods=["POST"])
def generate_high_res():
    # Loading the model 
    # Define the path to the checkpoint file
    checkpoint_path = r"C:\Users\dell\Downloads\checkpoint_epoch_830.pth"

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Extract generator state dictionary from the checkpoint
    gen_state_dict = checkpoint['gen_state_dict']

    # Instantiate the generator using the defined class
    # Assuming you have defined the Generator class with 'num_channels' and 'num_blocks'
    gen = Generator(in_channels=num_channels, num_blocks=1)
    gen.load_state_dict(gen_state_dict) 
    gen.eval()

    # Define transformations to convert tensors to PIL images
    to_pil = transforms.ToPILImage()
    # Receive low-res image from request
    if 'file' not in request.files:
        return "No file part"
    low_res_img = request.files['file']
    if low_res_img.filename == '':
        return "No selected file"
    if low_res_img:
        # Convert received image to PIL format
        low_res_pil = Image.open(low_res_img).convert('RGB')

        # Convert low-res image to tensor
        transform = transforms.Compose([
            transforms.Resize((256 , 256)),  # Assuming the low-res image size is 64x64
            transforms.ToTensor()
        ])
        low_res_tensor = transform(low_res_pil).unsqueeze(0)

        # Generate high-res image from low-res input
        with torch.no_grad():
            fake_high_res = gen(low_res_tensor)
        fake_high_res_pil = to_pil(fake_high_res[0].detach().cpu())

        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        fake_high_res_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Define the path where you want to save the image
        output_path = r"C:\Users\dell\Downloads\output_high_res.png"
        
        # Save the image to the specified path
        with open(output_path, "wb") as f:
            f.write(img_bytes.getvalue())

        # Return high-res image in the response body
        return Response(img_bytes, mimetype='image/png')

    else:
        return "Error in generating high-res image"

if __name__ == '__main__':
    app.run(debug=True)