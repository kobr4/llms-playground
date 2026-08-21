import qwen25_cpu

def classify(prompt):
    prompt = f"""How do you classifiy this sentence regarding your role as AI assistant with screenshot capabilities: "{prompt}" between:
    - Assistant should take a screenshot : return "#capture"
    - Assistant shoud write computer code : return "#code"
    - Assistant should generate an image : return "#image"
    - Normal sentence : return "#normal"
    Only return the selected hashtag: #capture, #code, #image or #normal
    """

    messages = [
        {"role": "system", "content": "Your an AI assistant and provide useful information to user."},
        {"role": "user", "content": prompt},
    ]

    hashtag = qwen25_cpu.predict(messages)
    return hashtag.replace('"',"")

def generate_filename(prompt):
    prompt = f"""Generate a filename from this prompt : "{prompt}". Only return the filename."""

    messages = [
        {"role": "system", "content": "Your an AI assistant and provide useful information to user."},
        {"role": "user", "content": prompt},
    ]

    filename = qwen25_cpu.predict(messages)
    return filename.replace('"',"")