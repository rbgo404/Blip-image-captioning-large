import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class InferlessPythonModel:
    def initialize(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16,device_map="cuda")

    def infer(self, inputs):
        img_url = inputs["img_url"]
        text = inputs.get("text")
        
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = self.processor(raw_image,text, return_tensors="pt").to("cuda", torch.float16)     
        out = self.model.generate(**inputs)
        generated_output = self.processor.decode(out[0], skip_special_tokens=True)
        
        return {'generated_output': generated_output}

    def finalize(self):
        self.model = None
