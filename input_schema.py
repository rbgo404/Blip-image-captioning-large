INPUT_SCHEMA = {
    "img_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"]
    },
    "text": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["a photography of"]
    }
}
